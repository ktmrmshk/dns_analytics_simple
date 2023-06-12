# Databricks notebook source
# MAGIC %md # 事前設定

# COMMAND ----------

# MAGIC %md ##パラメータ設定

# COMMAND ----------

dbfs_dir = '/tmp/your_name/dns_analytics'
dbname = 'your_name_db'

# COMMAND ----------

# code for Cleanup
spark.sql(f'DROP DATABASE {dbname} CASCADE')
dbutils.fs.rm(dbfs_dir, True)

spark.sql(f'CREATE DATABASE IF NOT EXISTS {dbname}')
spark.sql(f'USE {dbname}')

print(f'working dir on dbfs: {dbfs_dir}')
print(f'database name: {dbname}')

# COMMAND ----------

# MAGIC %md ## データセットの配置

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /tmp
# MAGIC rm -r dns-notebook-datasets
# MAGIC git clone https://github.com/zaferbil/dns-notebook-datasets.git

# COMMAND ----------

dbutils.fs.cp('file:///tmp/dns-notebook-datasets/data', f'{dbfs_dir}/datasets/', True)
dbutils.fs.cp('file:///tmp/dns-notebook-datasets/model', f'{dbfs_dir}/model/', True)

# COMMAND ----------

# MAGIC %md # 1. データの読み込み、Deltaテーブル化

# COMMAND ----------

# MAGIC %md ## 1.1 パッシブDNSログ
# MAGIC
# MAGIC `dns_events.json` => `bronze_dns`テーブル

# COMMAND ----------

from pyspark.sql.functions import *

pdns_schema = '''
  rrname     string,
  rrtype     string,
  time_first long,
  time_last  long,
  count      long,
  bailiwick  string,
  rdata      array<string>
'''

# jsonの読み込み
df = (
    spark.read.format('json')
    .schema(pdns_schema)
    .load(f'{dbfs_dir}/datasets/dns_events.json')
)

# カラムを追加
df_cleaned = (
    df.withColumn('rdatastr', concat_ws(',', col('rdata')))
)

# Deltaテーブルとして保存(永続化)
(
    df_cleaned.write.format('delta')
    .mode('overwrite')
    .option('mergeSchema', True)
    .saveAsTable('bronze_dns')
)

# 確認
display(
    spark.read.table('bronze_dns')
)

# COMMAND ----------

# MAGIC %md ## 1.2 URLHaus threat feedテーブル
# MAGIC
# MAGIC `ThreatDataFeed.txt` => `silver_threat_feeds`テーブル
# MAGIC
# MAGIC [URLHaus](https://urlhaus.abuse.ch/): マルウェアのURLをリストで公開

# COMMAND ----------

# MAGIC %pip install tldextract dnstwist geoip2

# COMMAND ----------

import tldextract
import numpy as np

def registered_domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return " "
    else:
        return ext.registered_domain

@udf
def domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return " "
    else:
        return ext.domain

#SQLからの参照のため、registerする
spark.udf.register('domain_extract', domain_extract)

df_threat_feed = (
    spark.read.format('csv')
    .option('header', True)
    .load(f'{dbfs_dir}/datasets/ThreatDataFeed.txt')
)

df_threat_feed_cleaned = (
    df_threat_feed
    .withColumn('domain', domain_extract('url'))
    .filter('char_length(domain) >= 2')
)

# Deltaテーブルに書き込む
(
    df_threat_feed_cleaned
    .write.format('delta')
    .mode('overwrite')
    .option('mergeSchema', True)
    .saveAsTable('silver_threat_feeds')
)

# 確認
display(
    spark.read.table('silver_threat_feeds')
)

# COMMAND ----------

# MAGIC %md ## 1.3 DNS Twistテーブル
# MAGIC
# MAGIC `dnstwist`ツールで酷似ドメインを生成
# MAGIC
# MAGIC `domains_dnstwists.csv` => `silver_twisted_domain_brand` テーブル

# COMMAND ----------

# 読み込み
df_twist = (
    spark.read.format('csv')
    .option('header', True)
    .load(f'{dbfs_dir}/datasets/domains_dnstwists.csv')
)

# 簡単に加工
df_twist_cleaned = (
    df_twist
    .withColumn('dnstwisted_domain', domain_extract('domain'))
    .filter('char_length(dnstwisted_domain) >= 2')
)

# Deltaテーブル化
(
    df_twist_cleaned.write.format('delta')
    .mode('overwrite')
    .option('mergeSchema', True)
    .saveAsTable('silver_twisted_domain_brand')
)

# 確認
display(
    spark.read.table('silver_twisted_domain_brand')
)

# COMMAND ----------

# MAGIC %md # 関連ツールの連携

# COMMAND ----------

# MAGIC %md ## GeoIP

# COMMAND ----------

import geoip2.errors
from geoip2 import database

import pandas as pd

from pyspark.sql.functions import pandas_udf
from pyspark import SparkFiles

# ipからロケーション情報を返す
def extract_geoip_data(ip: str, geocity):
    print(ip)
    if ip:
        try:
            record = geocity.city(ip)
            return {'city': record.city.name, 'country': record.country.name, 'country_code': record.country.iso_code}
        except geoip2.errors.AddressNotFoundError:
            pass
    return {'city': None, 'country': None, 'country_code': None}


@pandas_udf("city string, country string, country_code string")
def get_geoip_data(ips: pd.Series) -> pd.DataFrame:
    geocity = database.Reader(f'/dbfs{dbfs_dir}/datasets/GeoLite2_City.mmdb')
    extracted = ips.apply(lambda ip: extract_geoip_data(ip, geocity))
  
    return pd.DataFrame(extracted.values.tolist())

# COMMAND ----------

# MAGIC %md ## 学習済みのIoC判定モデル
# MAGIC
# MAGIC ドメイン名からIoCか正常ドメインかを判定するモデル(ランダムフォレスト、学習方法はAppendixで実施)

# COMMAND ----------

import mlflow
import mlflow.pyfunc

loaded_model = mlflow.pyfunc.load_model(f'dbfs:{dbfs_dir}/model')
ioc_detect_udf = spark.udf.register("ioc_detect", loaded_model.predict)

# COMMAND ----------

# MAGIC %md # ツールの適用
# MAGIC
# MAGIC パッシブDNSログデータ(Bronze)に対して、Geoロケーション情報とIoC判定を実施
# MAGIC
# MAGIC `bronze_dns` => `silver_dns`テーブル

# COMMAND ----------

# bronze_dnsテーブルからデータを加工
df_dns_table = spark.sql('''
    SELECT
        *,
        CASE WHEN rrtype = 'A' THEN element_at(rdata, 1) 
            ELSE null 
        END as ip_address
    FROM bronze_dns                       
''')

# 引き続き、加工
df_dns_table_cleaned = (
    df_dns_table
    .withColumn('geoip_data', get_geoip_data(df_dns_table.ip_address))
    .selectExpr(
        '*',
        'geoip_data.*',
        'CASE WHEN char_length(domain_extract(rrname)) > 5 \
             THEN ioc_detect(string(domain_extract(rrname))) ELSE null END AS ioc',
        'domain_extract(rrname) AS domain_name'
        )
    .drop('geoip_data')
)

# Deltaに永続化
(
    df_dns_table_cleaned
    .write.format('delta')
    .mode('overwrite')
    .option('mergeSchema', True)
    .saveAsTable('silver_dns')
)

# 確認
display(
    spark.read.table('silver_dns')
)

# COMMAND ----------

# MAGIC %md # アドホック分析

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DNSログの中でユニークなドメイン名の数
# MAGIC SELECT count(distinct(domain_name)) FROM silver_dns

# COMMAND ----------

# MAGIC %sql
# MAGIC -- silver_dnsのテーブルサンプル
# MAGIC SELECT * FROM silver_dns limit 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ロケーション
# MAGIC SELECT * FROM silver_dns WHERE ioc = 'ioc'

# COMMAND ----------

# MAGIC %md ### 分析例
# MAGIC
# MAGIC 以下の条件を全て満たすようなドメインとロケーションを特定する。
# MAGIC
# MAGIC > * 機械学習モデルからIoCと判定されたレコード
# MAGIC > * ドメイン名に`ip`の文字を含まない
# MAGIC > * ドメインの長さが8文字以上
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*), domain_name, country 
# MAGIC   from silver_dns 
# MAGIC   where ioc = 'ioc' and domain_name not like '%ip%' and char_length(domain_name) > 8 
# MAGIC   group by domain_name, country 
# MAGIC   order by count(*) desc

# COMMAND ----------

# MAGIC %md ### 分析例
# MAGIC
# MAGIC DNSログの中で、URLHausから提供された脅威リストに該当するドメイン
# MAGIC
# MAGIC (２つのテーブルを結合し、検索する)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT distinct(domain_name)
# MAGIC FROM silver_dns, silver_threat_feeds
# MAGIC WHERE silver_dns.domain_name == silver_threat_feeds.domain

# COMMAND ----------

# MAGIC %md ### 分析例
# MAGIC
# MAGIC 複数のテーブルの結合結果から、以下の条件を満たすレコードを検索
# MAGIC
# MAGIC > * DNSログの中で、URLHausから提供された脅威リストに該当するドメイン
# MAGIC > * ドメインの`rrname`が `ns1.asdklgb.cf.` もしくは `cn.`で終わる
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Looking for specific rrnames in multiple tables.
# MAGIC select
# MAGIC   domain_name,
# MAGIC   rrname,
# MAGIC   country,
# MAGIC   time_first,
# MAGIC   time_last,
# MAGIC   ioc,
# MAGIC   rrtype,
# MAGIC   rdata,
# MAGIC   bailiwick,
# MAGIC   silver_threat_feeds.*
# MAGIC from
# MAGIC   silver_dns,
# MAGIC   silver_threat_feeds
# MAGIC where
# MAGIC   silver_dns.domain_name == silver_threat_feeds.domain
# MAGIC   and (
# MAGIC     silver_dns.rrname = "ns1.asdklgb.cf."
# MAGIC     OR silver_dns.rrname LIKE "%cn."
# MAGIC   )

# COMMAND ----------

# MAGIC %md # ストリーミングデータに機械学習を適用し、IoCの判定を行う
# MAGIC
# MAGIC 逐次入ってくるログデータに対して、リアルタイムでIoC検知を実施する。

# COMMAND ----------

# (再掲)
# 機械学習モデルの読み込み

import mlflow
import mlflow.pyfunc

loaded_model = mlflow.pyfunc.load_model(f'dbfs:{dbfs_dir}/model')
ioc_detect_udf = spark.udf.register("ioc_detect", loaded_model.predict)


# COMMAND ----------

# MAGIC %md ## ログをストリーミングで受信し、機械学習モデルを当てる

# COMMAND ----------

pdns_schema = """
  rrname     string,
  rrtype     string,
  time_first long,
  time_last  long,
  count      long,
  bailiwick  string,
  rdata      array<string>
"""

# 読み込み
df = (
    spark.readStream
    .option('maxFilesPerTrigger', 1)
    .json(f'{dbfs_dir}/datasets/latest/', schema=pdns_schema)
    .withColumn('isioc', ioc_detect_udf(domain_extract('rrname')))
    .withColumn('domain', domain_extract('rrname'))
)

# Cleanup
chkpoint = f'{dbfs_dir}/checkpoint'
dbutils.fs.rm(chkpoint, True)

# Deltaへ結果を書き込む
(
    df.writeStream.format('delta')
    .outputMode('append')
    .option('checkpointLocation', chkpoint)
    .toTable('ioc_monitor')

)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ストリーミングでモニタする
# MAGIC SELECT * FROM ioc_monitor WHERE isioc = 'ioc'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT silver_twisted_domain_brand.*  FROM ioc_monitor, silver_twisted_domain_brand 
# MAGIC WHERE silver_twisted_domain_brand.dnstwisted_domain = ioc_monitor.domain

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from silver_threat_feeds 
# MAGIC where silver_threat_feeds.domain = domain_extract('ns1.asdklgb.cf.')

# COMMAND ----------

# MAGIC %md # Appendix: IoC判定の機械学習モデルの作成

# COMMAND ----------


