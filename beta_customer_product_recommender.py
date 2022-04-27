from re import L
import pyspark
##### function to create all tables
from pyspark.sql.types import *
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
from pyspark.sql.functions import first
from pyspark.sql.functions  import date_format
from pyspark.sql.functions import lit,StringType

from pyspark.sql.functions import udf,trim, upper, to_date, substring, length, min, when, format_number, dayofmonth, hour, dayofyear,  month, year, weekofyear, date_format, unix_timestamp
from pyspark import SparkConf
from pyspark.sql.functions import coalesce
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.functions import UserDefinedFunction
import datetime
from pyspark.sql.functions import year
from pyspark.sql.functions import datediff,coalesce,lag
from pyspark.sql.functions import when, to_date
from pyspark.sql.functions import date_add
from pyspark.sql.functions import UserDefinedFunction
import argparse
import traceback
import sys
import time
import math
from datetime import datetime

# functions

# create string index column for non-numeric id columns
def add_string_index(df,index_cols):


    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer

    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").setHandleInvalid("skip").fit(df) for column in index_cols ]


    pipeline = Pipeline(stages=indexers)
    df_r = pipeline.fit(df).transform(df)

    # df_r[['producttype','producttype_index','tz_brandname','tz_brandname_index']].show()

    return df_r

# function to train model and create recommendations, one store at a time
def customer_product_rec(str_id,ticket_df,filter_date,tz_product_df,exp_name):

    # test one store filtering for just airfield
    ticket_df=ticket_df[ticket_df['storeid']==str_id]

    df=ticket_df[ticket_df['storeid']!=230][['tz_product_id','customer_uuid','qty','dateclosed']]

    df=df[df['qty']>0.0]

    df=df[df['tz_product_id'].isNotNull()]

    # adding current date
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    df = df.withColumn('current_date',to_date(unix_timestamp(lit(timestamp),'yyyy-MM-dd').cast("timestamp")))

    # testing since frozen data only goes up to 2022-02-22
    df=df[df['dateclosed']>filter_date]

    # getting total qty of product purchased by store during period
    sum_df=df.groupBy('customer_uuid').agg({'qty':'sum'})\
    .select(col('customer_uuid'),
           col('sum(qty)').alias('total_qty'))

    # total qty of product purchased by store during period
    df=df.groupBy(['customer_uuid','tz_product_id']).agg({'qty':'sum'})\
    .select(col('customer_uuid'),
           col('tz_product_id'),
           col('sum(qty)').alias('product_qty')
           )


    df=df.alias('a')\
    .join(sum_df.alias('b'),
         (col('a.customer_uuid')==col('b.customer_uuid')),
          how='left'
         )\
    .select(
    col('a.tz_product_id'),
    coalesce(col('a.product_qty')/col('total_qty'),lit(0)).alias('norm_qty'),
    col('a.customer_uuid')
    )


    data=df


    test_df=df

    test_df2=add_string_index(df=test_df,index_cols=['tz_product_id','customer_uuid'])


    data=test_df2

    # splitting data for training and testing
    (training, test) = data.randomSplit([0.7, 0.3], seed = 2020)

    # write out datassets

    data.write.mode("overwrite")\
    .parquet('s3://treez-analysis-test/customer_product_recommendation/full_data/'+exp_name+'/'+str(str_id))

    training.write.mode("overwrite")\
    .parquet('s3://treez-analysis-test/customer_product_recommendation/train_data/'+exp_name+'/'+str(str_id))

    test.write.mode("overwrite")\
    .parquet('s3://treez-analysis-test/customer_product_recommendation/test_data/'+exp_name+'/'+str(str_id))


    frozen_training=sqlContext.read.parquet('s3://treez-analysis-test/customer_product_recommendation/train_data/'+exp_name+'/'+str(str_id))



    # Build the recommendation model using ALS on the training data
    als = ALS(implicitPrefs=True,
    nonnegative = True,
    rank=20,
    alpha=1,
    maxIter=10,
    regParam=0.1,
              userCol="customer_uuid_index",
              itemCol="tz_product_id_index",
              ratingCol="norm_qty",
             coldStartStrategy="drop")
    model = als.fit(frozen_training)

    # save trained model

    model.write().overwrite().save("s3://treez-analysis-test/trained_models/customer_product_recommender"+exp_name+'/'+str(str_id)+".model")



    # read in frozen data

    frozen_data=sqlContext.read.parquet('s3://treez-analysis-test/customer_product_recommendation/full_data/'+exp_name+'/'+str(str_id))

    # getting predictions for all stores

    all_customers_predictions=model.transform(test).alias('a')\
    .join(ticket_df[['tz_product_id','tz_productname','master_product_id']].distinct().alias('b'),
         (col('a.tz_product_id')==col('b.tz_product_id')),
          how='inner'
         )\
    .select(col('a.*'),
           col('b.tz_productname'),
           col('b.master_product_id'))



    # all_customers_predictions.show(truncate=False)

    all_customers_predictions.write.mode("overwrite")\
    .parquet('s3://treez-analysis-test/customer_product_recommendation/predictions/'+exp_name+'/'+str(str_id))


    # write out top 5 customer recs

    userRecs = model.recommendForAllUsers(5)

    userRecs.write.mode("overwrite")\
    .parquet('s3://treez-analysis-test/customer_product_recommendation/recs5/'+exp_name+'/'+str(str_id))

    # write out top 5 customer ratings

    prodRecs = model.recommendForAllItems(5)

    prodRecs.write.mode("overwrite")\
    .parquet('s3://treez-analysis-test/customer_product_recommendation/ratings5/'+exp_name+'/'+str(str_id))


    # code below adapted from sample_cust_eval function

    # adding product attributes to data and formatting recommendations

    top5_df=userRecs.alias('a').join(frozen_data[['customer_uuid_index','customer_uuid']]\
          .distinct().alias('b'),
         col('a.customer_uuid_index')==col('b.customer_uuid_index'),
          how='inner'
         )\
    .select(
    col('a.*'),
        col('b.customer_uuid')
    )\
    .select('customer_uuid',explode('recommendations').alias('recs'))\
    .select('customer_uuid','recs.*')


    prod_index_df=ticket_df[['tz_product_id','master_product_id']].distinct().alias('a')\
        .join(frozen_data[['tz_product_id','tz_product_id_index']].distinct().alias('b'),
             col('a.tz_product_id')==col('b.tz_product_id'),
              how='left'
             )\
        .select(
        col('a.tz_product_id'),
            col('a.master_product_id'),
            col('b.tz_product_id_index')
        )

    cust_index_df=all_customers_predictions[['customer_uuid','customer_uuid_index']].distinct()


    sample5_recs=top5_df.alias('a')\
    .join(prod_index_df.alias('b'),
         col('a.tz_product_id_index')==col('b.tz_product_id_index'),
          how='left'
         )\
    .select(col('a.*'),
           col('b.master_product_id'))\
    .alias('c')\
    .join(tz_product_df.alias('d'),
          col('c.master_product_id')==col('d.product_id'),
          how='left'
    )\
    .select(
    col('c.*'),
        coalesce(col('d.tz_productname'),col('d.productname')).alias('tz_productname'),
        coalesce(col('d.tz_productbrand'),col('d.productbrand')).alias('tz_productbrand'),
        coalesce(col('d.tz_producttype'),col('d.producttype')).alias('tz_producttype'),
        col('d.tz_size'),
        col('d.tz_classification'),
        col('d.tz_productsubtype'),
        col('d.tz_total_mg_cbd'),
        col('d.tz_total_mg_thc'),
        col('d.tz_amount'),
        col('d.tz_unitofmeasure'),
        col('d.productname'),
        col('d.productbrand'),
        col('d.producttype')
    )\
    .sort('customer_uuid','rating',ascending=False)

    return sample5_recs.withColumn('storeid',lit(str(str_id)))


def main(exp_name, storeid):
    # read in tables

    ticket_df=sqlContext.read.parquet('s3://treez-data-lake/daily_sku_validation_022322/ticket/')
    tz_product_df=sqlContext.read.parquet('s3://treez-data-lake/daily_sku_validation_022322/tz_product/')
    hourly_remaining_df=sqlContext.read.parquet('s3://treez-artifacts/hourly_remaining_units/')

    ticket_df.createOrReplaceTempView("ticket")
    tz_product_df.createOrReplaceTempView("tz_product")
    hourly_remaining_df.createOrReplaceTempView('hourly_remaining_units')

    subset_query="""select a.*
    from ticket a
    join hourly_remaining_units b
    on a.product_id=b.product_id
    and a.storeid = b.storeid
    and b.sellable_hourly_product_remaining > 0
    and b.hourly_product_remaining > 0
    """
    instock_df=spark.sql(subset_query)
    customer_product_rec(str_id=str(storeid),
                                     ticket_df=instock_df,
                                     filter_date='2021-02-21',
                                     tz_product_df=tz_product_df,
                                    exp_name=exp_name).write.mode("overwrite")\
                                .parquet('s3://treez-analysis-test/customer_product_recommendation/'+exp_name + '/' + str(storeid))

    print(f'Recommendations created for : {storeid}')
    print('s3://treez-analysis-test/customer_product_recommendation/'+exp_name + '/' + str(storeid))



if __name__  == '__main__':
    # get args
    parser = argparse.ArgumentParser(
    description='Create product recommendations')
    parser.add_argument('-id', '--storeid',
                        help='Stored id to create recommendations for.', required=True)
    parser.add_argument('-name', '--exp',
                        help='Name of the table', required=True)

    args = parser.parse_args()

    # create spark config
    conf = pyspark.SparkConf()
    spark = SparkSession.builder \
                .appName("beta_customer_product_recommender") \
                .config('spark.sql.codegen.wholeStage', False) \
                .getOrCreate()
    sc = SparkContext.getOrCreate(conf=conf)
    sqlContext = SQLContext(sc)

    # need to turn this into if statement for dev or prod base_path

    base_path='s3://treez-data-lake/'

    main(storeid=args.storeid, exp_name=args.exp)
