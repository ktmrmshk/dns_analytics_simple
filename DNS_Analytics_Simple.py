# Databricks notebook source
# MAGIC %md # 事前設定

# COMMAND ----------

# MAGIC %pip install tldextract dnstwist geoip2

# COMMAND ----------

# MAGIC %md ##パラメータ設定

# COMMAND ----------

dbfs_dir = '/tmp/your_name/dns_analytics'
dbname = 'your_name_db'

# COMMAND ----------

# code for Cleanup
spark.sql(f'DROP DATABASE {dbname} CASCADE')
dbutils.fs.rm(dbfs_dir, True)

# COMMAND ----------

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

# MAGIC %md # 2. 関連ツールの連携

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

# MAGIC %md ## ツールの適用
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

# MAGIC %md # 3. アドホック分析

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

# MAGIC %md # 4. ストリーミングデータに機械学習を適用し、IoCの判定を行う
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
# MAGIC
# MAGIC ### 全体の方針
# MAGIC
# MAGIC 1. 学習に用いるデータの準備
# MAGIC 1. 特徴量エンジニアリング
# MAGIC 1. 学習
# MAGIC 1. 評価・ハイパーパラメータチューニング(今回はシンプルのため、この過程をスキップ)
# MAGIC 1. データに適用(判定、`predict()`)

# COMMAND ----------

# MAGIC %md ### 1. 学習に用いるデータの準備
# MAGIC * 正規(Legit)のドメインリスト(Alexa)
# MAGIC * 不正(IoC)のドメインリスト(DGA)

# COMMAND ----------

import pandas as pd
import tldextract
import numpy as np

def registered_domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return " "
    else:
        return ext.registered_domain

def domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return " "
    else:
        return ext.domain


# 正常なドメイン
pdf_alexa = pd.read_csv(f'/dbfs{dbfs_dir}/datasets/alexa_100k.txt')

# ドメインのコア部分を抜く
pdf_alexa = pd.DataFrame( [domain_extract(uri) for uri in pdf_alexa['uri'] ], columns=['domain'] )

# クリーニング
pdf_alexa.dropna()
pdf_alexa.drop_duplicates()

# 正常(legit)のタグをつける
pdf_alexa['class'] = 'legit'

# ランダムshuffle(トレーニングのため)
pdf_alexa = pdf_alexa.reindex(np.random.permutation(pdf_alexa.index))

# 確認
display(pdf_alexa)

# COMMAND ----------

# IoCのドメイン(DGA: ドメイン生成アルゴリズム)
pdf_dga = pd.read_csv(f'/dbfs{dbfs_dir}/datasets/dga_domains_header.txt')

pdf_dga['domain'] = pdf_dga.applymap(lambda x: x.split('.')[0].strip().lower())

# クリーニング
pdf_dga.dropna()
pdf_dga.drop_duplicates()

# dga検出(ioc)のタグをつける
pdf_dga['class'] = 'ioc'

# ランダムshuffle(トレーニングのため)
pdf_dga = pdf_dga.reindex(np.random.permutation(pdf_dga.index))

# 確認
display(pdf_dga)

# COMMAND ----------

# 正常ドメインと、DGAドメインを一つのDataFrameにまとめる
pdf_all_domain = pd.concat([pdf_alexa, pdf_dga], ignore_index=True)

# 確認
display(pdf_all_domain)

# COMMAND ----------

# MAGIC %md ### 2. 特徴量エンジニアリング
# MAGIC
# MAGIC * 特徴量1: ドメインの文字列数
# MAGIC * 特徴量2: 文字列のエントロピー(どれだけ文字列が分散的に構成されているか)
# MAGIC * 特徴量3: 一般ドメイン(Alexa)のn-gramベクタとの類似度(内積)
# MAGIC * 特徴量4: 辞書データのngramベクタとの類似度(内積)
# MAGIC

# COMMAND ----------

## 特徴量1: 文字列の長さ
pdf_all_domain['length'] = [len(x) for x in pdf_all_domain['domain']]

## 特徴量2: エントロピー
import math
from collections import Counter
 
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values()) 

pdf_all_domain['entropy'] = [ entropy(x) for x in pdf_all_domain['domain']]


# COMMAND ----------

# 特徴量1, 2でクラスごとの分布を俯瞰する
def plot_hist(pdf, col, class1, class2, title):
    import numpy as np
    import matplotlib.pyplot as plt

    a=pdf[pdf_all_domain['class'] == class1][col]
    b=pdf[pdf_all_domain['class'] == class2][col]

    fig=plt.figure()
    fig1 = fig.add_subplot()
    fig2 = fig1.twinx()

    fig1.hist(a, bins=50, alpha=0.5, color='red', label=class1)
    fig2.hist(b, bins=50, alpha=0.5, color='blue',label=class2)

    h1, l1 = fig1.get_legend_handles_labels()
    h2, l2 = fig2.get_legend_handles_labels()

    fig1.set_title(title)
    fig1.legend(h1+h2, l1+l2, loc='upper left')


plot_hist(pdf_all_domain, 'length', 'legit', 'ioc', 'length')
plot_hist(pdf_all_domain, 'entropy', 'legit', 'ioc', 'entropy')


# COMMAND ----------

## 特徴量3 一般ドメイン(Alexa)のn-gramベクタとの類似度(内積)
## 特徴量4 辞書ワードのn-gramベクタとの類似度(内積)

import sklearn.ensemble
from sklearn import feature_extraction

# 特徴量3のngramベクタを用意
alexa_vec = sklearn.feature_extraction.text.CountVectorizer(
    analyzer='char',
    ngram_range=(3,5), 
    min_df=1e-4,
    max_df=1.0
)

cnt_matrix = alexa_vec.fit_transform(pdf_alexa['domain'])
alexa_cnt_vec = np.log10(cnt_matrix.sum(axis=0).getA1())
alexa_cnt_vec_name = alexa_vec.get_feature_names()


# 特徴量4のngramベクタを用意
pdf_dict_words = pd.read_csv(f'/dbfs{dbfs_dir}/datasets/words.txt', header=0, sep=';')
pdf_dict_words = pdf_dict_words[pdf_dict_words['words'].map(lambda x: str(x).isalpha())] #アルファベットのみ
pdf_dict_words = pdf_dict_words.applymap(lambda x: str(x).strip().lower()) #小文字に統合
pdf_dict_words = pdf_dict_words.dropna()
pdf_dict_words = pdf_dict_words.drop_duplicates()

dict_vec = sklearn.feature_extraction.text.CountVectorizer(
    analyzer='char', 
    ngram_range=(3,5), 
    min_df=1e-5, 
    max_df=1.0
)

cnt_matrix = dict_vec.fit_transform(pdf_dict_words['words'])
dict_cnt_vec = np.log10(cnt_matrix.sum(axis=0).getA1())
dict_cnt_vec_name = dict_vec.get_feature_names()

# 特徴量3と4を返す関数(それぞれの内積を返す関数) - demoのために使用
def ngram_count(domain):
    alexa_match = alexa_cnt_vec * alexa_vec.transform([domain]).T
    dict_match  = dict_cnt_vec  * dict_vec.transform([domain]).T
    print(f'Domain: {domain}, Alexa match: {alexa_match}, Dict match: {dict_match}')

# 例
ngram_count('beyonce')
ngram_count('dominos')
ngram_count('1cb8a5f36f')
ngram_count('zfjknuh38231')
ngram_count('bey6o4ce')
ngram_count('washington')


# COMMAND ----------

# 特徴量3と4をデータフレームに追加
pdf_all_domain['alexa_grams'] = alexa_cnt_vec * alexa_vec.transform(pdf_all_domain['domain']).T
pdf_all_domain['word_grams']  = dict_cnt_vec  * dict_vec.transform(pdf_all_domain['domain']).T

# 確認
display(pdf_all_domain)

# COMMAND ----------

# 特徴量3, 4でクラスごとの分布を俯瞰する

plot_hist(pdf_all_domain, 'alexa_grams', 'legit', 'ioc', 'alexa_grams')
plot_hist(pdf_all_domain, 'word_grams', 'legit', 'ioc', 'word_grams')


# COMMAND ----------

# MAGIC %md ### 3. 機械学習モデルの学習
# MAGIC
# MAGIC ここではシンプルにランダムフォレストで分類を実施する。
# MAGIC
# MAGIC * 説明変数: `length`, `entropy`, `alexa_grams`, `word_grams`
# MAGIC * 目的変数: `class`

# COMMAND ----------

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.model_selection import train_test_split

X = pdf_all_domain[ ['length', 'entropy', 'alexa_grams', 'word_grams'] ].values
y = pdf_all_domain['class'].values

with mlflow.start_run() as run:
    # MLflowで自動トラック
    mlflow.sklearn.autolog()

    # 学習データとテストデータを分ける。    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ランダムフォレスト分類器
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

# mlflowのRunIDを確認
run_id = run.info.run_id
print(f'MLflow run_id: {run_id}, model_uri: runs:/{run_id}/model')



# COMMAND ----------

# MAGIC %md ## 4. 本番用に学習したモデルを適用させる関数を定義

# COMMAND ----------

# 1. MLflowで扱うために、箱を作る(pyfunc)
import mlflow.pyfunc

class vec_transform(mlflow.pyfunc.PythonModel):
    def __init__(self, alexa_vec, dict_vec, ctx):
        self.alexa_vec = alexa_vec
        self.dict_vec = dict_vec
        self.ctx = ctx

    def predict(self, context, model_input):
        _alexa_match = alexa_cnt_vec * self.alexa_vec.transform([model_input]).T  
        _dict_match = dict_cnt_vec * self.dict_vec.transform([model_input]).T
        _X = [len(model_input), entropy(model_input), _alexa_match, _dict_match]
        return str(self.ctx.predict([_X])[0])
    

# 2. Mlflowのモデルとして整える
dbutils.fs.rm(f'{dbfs_dir}/models/my_ioc_detection_model', True)
my_model = vec_transform(alexa_vec, dict_vec, clf)
mlflow.pyfunc.save_model(f'/dbfs{dbfs_dir}/models/my_ioc_detection_model', python_model=my_model)



# COMMAND ----------

# 実際に判定してみる
my_model.predict(mlflow.pyfunc.PythonModel, '9aaaa0cmsdkaea')
