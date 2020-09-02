# ______________________imports________________________________

import gc
import time
import math
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyspark import ml
from pyspark.sql import Window
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql import SparkSession
spark = SparkSession.builder.enableHiveSupport().getOrCreate()
pd.set_option('max_columns',None)

# __________________set global variables________________________

RAW_DATA_PATH = 's3://prod-search-ranking-ml/experiments/wnd/India_POC/datapull_test'
REQUEST_PLACEID_PATH = 's3://prod-search-ranking-ml/experiments/wnd/request_placeid_1020_1130'
PLACEID_POPULARITY_PATH = 's3://prod-search-ranking-ml/experiments/wnd/location_popularity.csv'
LAT_LONG_PATH = 'locality_lat_long.csv'
HOTELS_PATH = 'hotels.csv'
START_DATE = 2019110100 #for test # 2019102000 # three months data is pulled to capture clickstreams but only one month of data is used for training the model
SAME_CITY_CLICKS = True

# columns to collect in clickstreams
COLLECTLIST_STR = ['hotelid','hotel_type']
COLLECTLIST_FLOAT = ['fprice','distance','rating']

# path to save intermediate data
INTERMEDIATE_DATA_PATH_1 = 's3://prod-search-ranking-ml/experiments/wnd/India_POC/Intermediate_1' 
INTERMEDIATE_DATA_PATH_2 = 's3://prod-search-ranking-ml/experiments/wnd/India_POC/Intermediate_2'
INTERMEDIATE_DATA_PATH_3 = 's3://prod-search-ranking-ml/experiments/wnd/India_POC/Intermediate_3'
INTERMEDIATE_DATA_PATH_4 = 's3://prod-search-ranking-ml/experiments/wnd/India_POC/Intermediate_4'

# Out path 
OUT_PATH = 's3://prod-search-ranking-ml/experiments/wnd/India_POC/dataprocess_test'

# ____________________ load datasets ___________________________
# load request placeid data 
request_placeid_df = spark.read.parquet(REQUEST_PLACEID_PATH).select(F.col('requestid'),
                                                                     F.col('search_place_id').alias('placeid'))
request_placeid_df = request_placeid_df.drop_duplicates(subset =['requestid'])

# load placeid and associated popularity
df_loc = spark.read.csv(PLACEID_POPULARITY_PATH,header = True).select(F.col('conversion'),
                                                                      F.col('popularity'),
                                                                      F.col('id').alias('placeid'))
df_loc = df_loc.drop_duplicates(['placeid'])
df_loc = df_loc.withColumn('place_popularity_score', 
                           F.when(F.isnull(F.col('conversion')), 
                           (2*100*0.0049)*(F.col('popularity')**0.5)).otherwise(
                           (2*100*F.col('conversion'))*(F.col('popularity')**0.5)))

# load raw data created using 01_datapull.py
raw_data = spark.read.parquet(RAW_DATA_PATH)
raw_data = raw_data.where('inserted_at >= "{}"'.format(START_DATE)).drop_duplicates().dropna(subset=['requestid','userid'])
raw_data = raw_data.where(F.col('nearbysearch') != 1) # nearby search is not considered in the model (assumption that nearby search is ranked based on distance)
raw_data = raw_data.where('userid not in (""," ")').where('userid is not null') # remove unidentified users (user history not available for these users)
raw_data = raw_data.drop('placeid').join(request_placeid_df,on = 'requestid', how='left') # replace placeid (old logic - owner : neel)
raw_data = raw_data.join(df_loc, on='placeid', how='left') # merge the place popularity score
raw_data = raw_data.withColumn('searchdate',F.from_unixtime(F.substring(F.col('searchtime'),0,10),'yyyy-MM-dd'))
raw_data = raw_data.withColumn('bookingFlag',F.when(F.col('event') == 'booking',1).otherwise(0)) # create the booking flag
raw_data = raw_data.withColumn('detailsFlag',F.when(F.col('event') == 'hotel_details_page',1).otherwise(0)) # create the details flag
raw_data = raw_data.withColumnRenamed('dispCategory','display_category')
# ____________________ create features __________________________
features = [
  F.col('userId'),
  F.col('fprice'),
  F.col('placeid'),
  F.col('hotelid'),
  F.col('distance'),
  F.col('requestid'),
  F.col('searchdate'),
  F.col('inserted_at'),
  F.col('hotel_type'),
  F.col('platform'),
  F.col('cityid'),
  F.col('dmodel'),
  F.col('rating'),
  F.col('rcount'), 
  F.col('rank'),
  F.col('user_abp'),
  F.col('iscitysearch'), 
  F.col('display_category'),              
  F.col('place_popularity_score'),
  F.col('searchtime'),
  F.coalesce(F.when(F.col('realised_flag').isNotNull(),'realised').otherwise(None),F.col('event')).alias('target'),
  F.when(F.col('hotelslasherprice') == 0,0).otherwise((F.col('hotelslasherprice') - F.col('fprice'))/F.col('hotelslasherprice')).alias('discount_per'),
  F.when(F.col('impressioncount') == 0,0).otherwise(F.col('bookingcount')/F.col('impressioncount')).alias('hotel_br'),
  F.when(F.col('impressioncount') == 0,0).otherwise(F.col('detailcount')/F.col('impressioncount')).alias('hotel_ctr'),
  F.when(F.col('detailcount') == 0,0).otherwise(F.col('bookingcount')/F.col('detailcount')).alias('hotel_btod'),
  F.when(F.col('userimpressioncount') == 0,0).otherwise(F.col('userbookingcount')/F.col('userimpressioncount')).alias('hotel_user_br'),
  F.when(F.col('userimpressioncount') == 0,0).otherwise(F.col('userdetailcount')/F.col('userimpressioncount')).alias('hotel_user_ctr'),
  F.when(F.col('userdetailcount') == 0,0).otherwise(F.col('userbookingcount')/F.col('userdetailcount')).alias('hotel_user_btod'),
  F.when(F.col('impression_count') == 0,0).otherwise(F.col('booking_count')/F.col('impression_count')).alias('avg_hotel_br'),
  F.when(F.col('impression_count') == 0,0).otherwise(F.col('details_count')/F.col('impression_count')).alias('avg_hotel_ctr'),
  F.when(F.col('details_count') == 0,0).otherwise(F.col('booking_count')/F.col('details_count')).alias('avg_hotel_btod'),
  F.when(F.datediff('checkout','checkin')>0,F.datediff('checkout','checkin')).otherwise(0).alias('stayLength'),
  F.when(F.dayofweek('checkin').isin([6,7,1]),1).otherwise(0).alias('is_weekend_checkin'),
  (F.dayofweek('checkin')).alias('day_of_week_checkin'),
  (F.col('fprice')-F.col('user_abp')).alias('abp_price_diff'),
  F.when(F.dayofweek('searchdate').isin([6,7,1]),1).otherwise(0).alias('is_weekend_searchDate'),
  F.dayofweek('searchdate').alias('day_of_week_searchDate'),
  F.when(F.datediff('searchdate','checkin')>0,F.datediff('searchdate','checkin')).otherwise(0).alias('advPurchaseWindow'),
  F.when(F.col('nearbysearch') == 1,1).otherwise(0).alias('is_nearBy_search'),
  F.when(F.col('iscitysearch') == 1,1).otherwise(0).alias('is_city_search'),
  F.when(F.col('distance') <= 5,1).otherwise(0).alias('vicinity'),
  F.col('rating').alias('ratingMean'),
  F.col('rcount').alias('ratingCount')
]

df_cleanbasic_training = raw_data.select(*features)

# missing value imputations
rating_mean_train = raw_data.groupby().agg(F.mean('rating')).collect()[0][0]
rating_count_mean_train = raw_data.groupby().agg(F.mean('rcount')).collect()[0][0]

df_cleanbasic_training = df_cleanbasic_training.fillna(0,subset=['iscitysearch','avg_hotel_rr',
                                                                 'bookingFlag','detailsFlag',
                                                                 'hotel_user_btod','hotel_user_ctr',
                                                                 'hotel_user_br','hotel_ctr',
                                                                 'hotel_btod','hotel_br','avg_user_rr',
                                                                 'avg_hotel_btod', 'avg_hotel_br',
                                                                 'avg_hotel_ctr','user_abp','iscitysearch'])
df_cleanbasic_training = df_cleanbasic_training.na.fill({'ratingMean':rating_mean_train,'ratingCount':rating_count_mean_train})

# remove request ids that did not result in a click 
train_requestids = raw_data.where('event is not null').select('requestid').drop_duplicates()
df_cleanbasic_training = df_cleanbasic_training.join(train_requestids,on='requestid',how='inner').repartition(800,'requestid')


# _____________________Create weighted features___________________________
# persist data for downstream iterative calculations
all_data = raw_data.repartition(800,'requestid')
all_data = all_data.persist()
all_data.count()

# method to create weighted average of a column 
def weightedAvg80(date_col,value_col):
  """
  computes 80th percentile of rolling weighted average given a global weighting matrix
  """
  zipped_data = sorted(list(zip(date_col,value_col)),key=lambda x: x[0])
  vals = np.array([i[1] for i in zipped_data])
  weighted_avg = np.dot(vals,multiplication_array)/sum_weights
  
  VT=np.array([vals for i in range(n_dates)]).T
  exploded_vals = np.square((VT - weighted_avg)*np.where(VT==0,0,1))
  weighted_var = np.sqrt(np.sum(exploded_vals*multiplication_array,axis=0))/sum_weights
  
  out_vals = weighted_avg - 1.282*weighted_var
  out_zipped = [[i[0] for i in zipped_data],out_vals.tolist()]
  return out_zipped
weightedAverageUdf = F.udf(weightedAvg80,returnType=T.ArrayType(elementType=T.ArrayType(T.StringType())))

# method to create weated average features of a df
def getWeightedAgg(all_data,key=['hotelid'],prefix='hotel'):
  """calculate weighter ctr, br and dtob for a dataframe
  """

  global multiplication_array,sum_weights,n_dates
  
  col_list = ['searchdate'] + key
  df_hotel = all_data.groupby(col_list).agg(F.sum('bookingFlag').alias('booking_count'),
                                            F.sum('detailsFlag').alias('details_count'),
                                            F.count('fprice').alias('impression_count'))
  
  # calculate the metrics to be weighted
  col_list_1 = col_list + [(F.col('details_count')/F.col('impression_count')).alias('raw_ctr'),
                           (F.col('booking_count')/F.col('impression_count')).alias('raw_br'),
                           (F.col('booking_count')/F.col('details_count')).alias('raw_dtob')]
  df_hotel = df_hotel.select(*col_list_1)
  
  # fill missing dates for all keys
  dates_df = df_hotel.select('searchdate').drop_duplicates()
  hotel_dates_df =  df_hotel.select(*key).drop_duplicates().crossJoin(dates_df)
  n_dates = dates_df.count()
  
  df_hotel = hotel_dates_df.join(df_hotel,on=col_list,how='left').fillna(0)

  # collect metrics as a list to pass through udf
  df_hotel = df_hotel.groupby(key).agg(F.collect_list('raw_ctr').alias('raw_ctr'),
                                       F.collect_list('raw_br').alias('raw_br'),
                                       F.collect_list('raw_dtob').alias('raw_dtob'),
                                       F.collect_list('searchdate').alias('searchdate'))
  
  
  # compute the global weighting matrix and sum across rows
  multiplication_array = np.arange(1,n_dates+1)
  multiplication_array = np.array([multiplication_array-i for i in range(0,n_dates)])
  multiplication_array = 8-multiplication_array
  multiplication_array = np.where(multiplication_array<=7,multiplication_array,0)
  multiplication_array = np.where(multiplication_array>=1,multiplication_array,0)
  sum_weights = multiplication_array.sum(axis=0)
  
  # calculate the rolling weighted average of metrics
  select_col = [F.col(i) for i in key] + [weightedAverageUdf(F.col('searchdate'),F.col('raw_ctr')).alias('ctr_array'),
                                          weightedAverageUdf(F.col('searchdate'),F.col('raw_br')).alias('br_array'),
                                          weightedAverageUdf(F.col('searchdate'),F.col('raw_dtob')).alias('dtob_array')]
  weighted_average_df = df_hotel.select(*select_col)
  
  select_col = [F.col(i) for i in key] + [F.arrays_zip(F.col('ctr_array').getItem(0).alias('searchdate'),
                                                       F.col('ctr_array').getItem(1).alias('ctr'),
                                                       F.col('br_array').getItem(1).alias('br'),
                                                       F.col('dtob_array').getItem(1).alias('dtob')).alias('zip_averages')]
  weighted_average_df = weighted_average_df.select(*select_col)
  
  select_col = [F.col(i) for i in key] + [F.explode('zip_averages').alias('averages')]
  weighted_average_df = weighted_average_df.select(*select_col)
  
  select_col = [F.col(i) for i in key] + [F.col('averages').getItem('0').alias('searchdate'),
                                          F.col('averages').getItem('1').alias(prefix + '_ctr'),
                                          F.col('averages').getItem('2').alias(prefix + '_br'),
                                          F.col('averages').getItem('3').alias(prefix + '_dtob')]
  weighted_average_df = weighted_average_df.select(*select_col)
  
  return weighted_average_df

# Weighted average calculation at a hotel, hotel+place, user + display cat level
df_hotel_events_avg = getWeightedAgg(all_data,key=['hotelid'],prefix='hotel')
df_hotel_place_events_avg = getWeightedAgg(all_data,key=['hotelid','placeid'],prefix='hotel_placeid')
df_user_cat_events_avg = getWeightedAgg(all_data, ['userId', 'display_category'], 'user_cat')

# merge wighted aggregates with data
df_cleanbasic_training = df_cleanbasic_training.drop('hotel_ctr','hotel_br','hotel_dtob').join(df_hotel_events_avg, on=['hotelid', 'searchdate'], how='left')
df_cleanbasic_training = df_cleanbasic_training.join(df_hotel_place_events_avg, on=['hotelid', 'placeid', 'searchdate'], how='left')
df_cleanbasic_training = df_cleanbasic_training.join(df_user_cat_events_avg, on=['userId', 'display_category', 'searchdate'], how='left')

df_cleanbasic_training = df_cleanbasic_training.withColumn('hotel_placeid_ctr',
                                                                F.when((F.col('iscitysearch') == 1) & (F.isnull(F.col('placeid'))),F.col('hotel_ctr')).\
                                                                otherwise(F.col('hotel_placeid_ctr')))

df_cleanbasic_training = df_cleanbasic_training.withColumn('hotel_placeid_br',
                                                                F.when((F.col('iscitysearch') == 1) & (F.isnull(F.col('placeid'))),F.col('hotel_br')).\
                                                                otherwise(F.col('hotel_placeid_br')))

df_cleanbasic_training = df_cleanbasic_training.withColumn('hotel_placeid_dtob',
                                                                F.when((F.col('iscitysearch') == 1) & (F.isnull(F.col('placeid'))),F.col('hotel_dtob')).\
                                                                otherwise(F.col('hotel_placeid_dtob')))

df_cleanbasic_training = df_cleanbasic_training.withColumn('is_locality_search',
                                                                F.when((F.col('iscitysearch') == 0) & (F.col('is_nearBy_search') == 0),F.lit(1)).\
                                                                 otherwise(F.lit(0))).repartition(800,'requestid')

df_cleanbasic_training.write.parquet(INTERMEDIATE_DATA_PATH_1)
df_cleanbasic_training = spark.read.parquet(INTERMEDIATE_DATA_PATH_1)
print(INTERMEDIATE_DATA_PATH_1 + 'Done!')

#__________________________ distance backfill for city search ________________________________


df_loc = pd.read_csv(LAT_LONG_PATH)
df_loc.sort_values(by=['conversion'], ascending=False, inplace=True)

df_hotel = pd.read_csv(HOTELS_PATH)
df_hotel.drop_duplicates(subset ="property_id", keep = 'first', inplace = True)

def rad(x):
  return x * math.pi / 180
  
def getRadialDist(hlat, hlng, llat, llng):
  r = 6371 # Earths mean radius in meter
  dlat = rad(hlat - llat)
  dlng = rad(hlng - llng)
  a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(rad(hlat)) * math.cos(rad(llat)) * math.sin(dlng/2) * math.sin(dlng/2)
  c = 2 * math.atan2(a**0.5, (1 - a)**0.5)
  d = r * c
  return d

def getHotelDistMap():  
  hotelDistMap = {}
  for hindex, hrows in df_hotel.iterrows():
    placeMap = {}
    for lindex, lrows in df_loc.iterrows():
      distance = getRadialDist(hrows.latitude, hrows.longitude, lrows.lat, lrows.lng)
      placeMap[lrows.id] = distance
    hotelDistMap[hrows.property_id] = placeMap
  return hotelDistMap

def getSortedPlaceIds_HotelIds():
  hotelids = df_hotel.property_id
  sortedPlaceids = df_loc.id.tolist()
  return sortedPlaceids, hotelids

def getHmapMinPopularDist():
    hotelDistMap = getHotelDistMap()
    sortedPlaceids, hotelids = getSortedPlaceIds_HotelIds()
    hMap = {}
    for i in hotelids:
        placeMap = hotelDistMap.get(i)
        dist = None
        listPlace = []
        listDist = []
        for key in placeMap:
          if (placeMap[key] <= 5):
            listPlace.append(key)
          listDist.append(placeMap[key])
        if (len(listPlace) > 0):
          indexList = []
          for k in listPlace:
            indexList.append(sortedPlaceids.index(k))
          indexList.sort()
          chosenPlaceId = sortedPlaceids[indexList[0]] 
          dist = hotelDistMap[i][chosenPlaceId]
        else:
          dist = (min(listDist) if len(listDist) > 0 else None)
        if (dist != None):
          hMap[str(i)] = dist
    return hMap

hMap = getHmapMinPopularDist()
hMap_df = spark.createDataFrame(pd.DataFrame(list(zip(hMap.keys(),hMap.values())),hMap,columns=['hotelid','distance_backfill'])).repartition(1)
hMap_df.show(10)

df_train_city_backfill = df_cleanbasic_training.join(F.broadcast(hMap_df),on='hotelid',how='left')
df_train_city_backfill = df_train_city_backfill.withColumn('distance_new',
                                                            F.when(F.col('iscitysearch') == 1,F.col('distance_backfill')).\
                                                            otherwise(F.col('distance'))).drop('distance','distance_backfill')
df_train_city_backfill = df_train_city_backfill.withColumnRenamed('distance_new','distance')

# clean the erronuous hotel listings in a search
hotel_city = spark.sql(""" select hotel_id as hotelid, city_id as cityid_hotel from aggregatedb.hotels_summary """).drop_duplicates()
df_train_city_backfill = df_train_city_backfill.join(hotel_city,on='hotelid',how='left')
df_train_city_backfill = df_train_city_backfill.where('cityid = cityid_hotel')

df_train_city_backfill.write.parquet(INTERMEDIATE_DATA_PATH_2)
print(INTERMEDIATE_DATA_PATH_2 + 'Done!')


#______________________clickstream features_________________________
# load raw data created using 01_datapull.py
df_train_data = spark.read.parquet(RAW_DATA_PATH)
df_train_data = df_train_data.drop_duplicates().dropna(subset=['requestid','userid'])
df_train_data = df_train_data.where(F.col('nearbysearch') != 1) # nearby search is not considered in the model (assumption that nearby search is ranked based on distance)
df_train_data = df_train_data.where('userid not in (""," ")').where('userid is not null') # remove unidentified users (user history not available for these users)
df_train_data = df_train_data.withColumn('searchdate',F.from_unixtime(F.substring(F.col('searchtime'),0,10),'yyyy-MM-dd'))

# filter for requestids where bookings are <=1
one_booking_request_df = df_train_data.groupby('requestid').agg(F.countDistinct('bookingid').alias('booking_flag_count'))
one_booking_request_df = one_booking_request_df.where('booking_flag_count < 2').select('requestid','booking_flag_count')
df_list_data = df_train_data.join(one_booking_request_df.select('requestid'), on = 'requestid', how = 'inner')

#filter out null events 
clicked_hotels = df_list_data.where('event is not null')

#create window object to collect past 20 clicks of a user 
window = Window.partitionBy('userid').orderBy('searchdate').rowsBetween(-20, -1) 

#method that filters out clicks from the candidate request id (current request id) from the collected list 
def filter_click(click_seq,request_seq,request_id,n=10):
    """filter out clicks from the same request id and return last n clicks
    """
    filtered_click = [i[0] for i in zip(click_seq,request_seq) if i[1] != request_id]
    filtered_click = filtered_click[-n:]
    return filtered_click

filter_click_udf_float = F.udf(filter_click,T.ArrayType(T.FloatType()))
filter_click_udf_str = F.udf(filter_click,T.ArrayType(T.StringType()))


# create column objects for the clickstreams 
column_list = ['requestid'] # initialise the list to use in the select method
clicked_hotels = clicked_hotels.withColumn('req_seq_10',F.collect_list('requestid').over(window)) # correct request id of the clicks 
for i in COLLECTLIST_STR: 
    column_list.append(filter_click_udf_str(F.collect_list(i).over(window),
                                            F.col('req_seq_10'),
                                            F.col('requestid'),F.lit(10)).alias('clicked_'+i+'_list_10'))
    
for i in COLLECTLIST_FLOAT:
    column_list.append(filter_click_udf_float(F.collect_list(i).over(window),
                                            F.col('req_seq_10'),
                                            F.col('requestid'),F.lit(10)).alias('clicked_'+i+'_list_10'))
    
collected_data = clicked_hotels.select(*column_list).drop_duplicates(subset=['requestid'])

# merge the clickstream with rest of the data
df_train_city_backfill = spark.read.parquet(INTERMEDIATE_DATA_PATH_2)
df_clickstream = df_train_city_backfill.join(collected_data,on='requestid',how='inner')
df_clickstream = df_clickstream.drop_duplicates(subset=['hotelid','requestid'])
df_clickstream.repartition(800,'requestid').write.parquet(INTERMEDIATE_DATA_PATH_3)
print(INTERMEDIATE_DATA_PATH_3 + 'Done!')

df_clickstream = spark.read.parquet(INTERMEDIATE_DATA_PATH_3)

# _____________mean, stddev and ctr calculation on clickstream data____________________

def ctr(val,list_):
    """method to calculate ctr from clickstream data
    """
    val = float(val)
    list_ = [float(i) for i in list_]
    try:
        list_filter = [i for i in list_ if i == val]
        return len(list_filter)/(len(list_) + 1e-8)
    except:
        return 0.0   
ctr_udf = F.udf(ctr,T.FloatType())


def average(x):
    """method to calculate mean from clicksteam data
    """
    try:
        mean_x = sum(x)/(len(x) + 1e-8)
    except:
        mean_x = None
    return mean_x
average_udf = F.udf(average,T.FloatType())

def stddev(data):
    """method to calculate stddev from clickstream data
    """
    try:
        c = sum(data)/(len(data) + 1e-8)
        ss = sum((x-c)**2 for x in data)
        pvar = ss/(len(data) + 1e-8)
        std_x = pvar**0.5
    except:
        std_x = None
    return std_x
stddev_udf = F.udf(stddev,T.FloatType())
    
# calculate ctr based on last 10 clicks
df_clickstream = df_clickstream.withColumn('hotel_clicked_rate',ctr_udf(F.col('hotelid'),F.col('clicked_hotelid_list_10')))
df_clickstream = df_clickstream.withColumn('cat_clicked_rate',ctr_udf(F.col('hotel_type'),F.col('clicked_hotel_type_list_10')))


original_columns = df_clickstream.columns # initialise column list
for i in ['clicked_hotelid_list_10','clicked_hotel_type_list_10','clicked_fprice_list_10','clicked_distance_list_10','clicked_rating_list_10']:
    original_columns.remove(i)

#convert hotelid and category clickstream list into columns
clicked_hotelid_columns = [F.col('clicked_hotelid_list_10').getItem(i).alias('clicked_hotelid_{}'.format(i+1)) for i in range(10)]
clicked_type_list_columns = [F.col('clicked_hotel_type_list_10').getItem(i).alias('clicked_hotel_type_{}'.format(i+1)) for i in range(10)]

select_columns = original_columns + clicked_hotelid_columns + clicked_type_list_columns

# calculate mean and std on last 10 clicks
for i in ['clicked_fprice_list_10','clicked_distance_list_10','clicked_rating_list_10']:
    select_columns.append(average_udf(i).alias(i))
    select_columns.append(stddev_udf(i).alias(i+'_std'))

df_clickstream = df_clickstream.select(select_columns)

# _________________________ Create additional features ________________________________

# feature indexing by other candidates in a requestid
reqest_mean = df_clickstream.groupby('requestid').agg({'fprice':'mean','rating':'mean','discount_per':'mean','distance':'mean'})
df_clickstream = df_clickstream.join(reqest_mean,on='requestid',how='left')

df_clickstream = df_clickstream.withColumn('fprice_ratio',F.col('fprice')/F.col('avg(fprice)'))
df_clickstream = df_clickstream.withColumn('rating_ratio',F.col('rating')/F.col('avg(rating)'))
df_clickstream = df_clickstream.withColumn('distance_ratio',F.col('distance')/F.col('avg(distance)'))
df_clickstream = df_clickstream.withColumn('discount_per_ratio',F.col('discount_per')/F.col('avg(discount_per)'))

df_clickstream = df_clickstream.drop('avg(fprice)','avg(rating)','avg(distance)','avg(discount_per)')

# create distance buckets
qds = ml.feature.QuantileDiscretizer(numBuckets=10, inputCol='distance', outputCol='distance_buckets', relativeError=0.001, handleInvalid=None)
df_preprocessed = qds.setHandleInvalid("keep").fit(df_clickstream).transform(df_clickstream)

df_preprocessed.repartition(800,'requestid').write.parquet(INTERMEDIATE_DATA_PATH_4)
print(INTERMEDIATE_DATA_PATH_4 + 'Done!')
df_preprocessed = spark.read.parquet(INTERMEDIATE_DATA_PATH_4)

# remove the ids from clickstream not present in the current set of hotel ids

hotel_idx = df_preprocessed.select(F.col('hotelid').alias('hotelid_sub'),F.col('cityid').alias('cityid_sub')).drop_duplicates().repartition(1)
hotel_type_idx = df_preprocessed.select(F.col('hotel_type').alias('hotel_type_sub'),F.col('cityid').alias('cityid_sub')).drop_duplicates().repartition(1)
cityid_idx = df_preprocessed.select(F.col('cityid').alias('cityid_sub')).drop_duplicates().repartition(1)

clicked_hotelid_cols = ['clicked_hotelid_{}'.format(i+1) for i in range(10)]  
clicked_type_list_cols = ['clicked_hotel_type_{}'.format(i+1) for i in range(10)]

if SAME_CITY_CLICKS:
    for i in clicked_hotelid_cols:
        df_preprocessed = df_preprocessed.join(F.broadcast(hotel_idx),(df_preprocessed[i]==hotel_idx.hotelid_sub)&(df_preprocessed['cityid']==hotel_idx.cityid_sub),'left')
        df_preprocessed = df_preprocessed.drop(i)
        df_preprocessed = df_preprocessed.drop('cityid_sub')
        df_preprocessed = df_preprocessed.withColumnRenamed('hotelid_sub',i).fillna(0,subset=[i])

    for i in clicked_type_list_cols:
        df_preprocessed = df_preprocessed.join(F.broadcast(hotel_type_idx),(df_preprocessed[i]==hotel_type_idx.hotel_type_sub)&(df_preprocessed['cityid']==hotel_type_idx.cityid_sub),'left')
        df_preprocessed = df_preprocessed.drop(i)
        df_preprocessed = df_preprocessed.drop('cityid_sub')
        df_preprocessed = df_preprocessed.withColumnRenamed('hotel_type_sub',i).fillna(0,subset=[i])
else:
    for i in clicked_hotelid_cols:
        df_preprocessed = df_preprocessed.join(F.broadcast(hotel_idx.select('hotelid_sub')),df_preprocessed[i]==hotel_idx.hotelid_sub,'left')
        df_preprocessed = df_preprocessed.drop(i)
        df_preprocessed = df_preprocessed.withColumnRenamed('hotelid_sub',i).fillna(0,subset=[i])
        
    for i in clicked_type_list_cols:
        df_preprocessed = df_preprocessed.join(F.broadcast(hotel_type_idx.select('hotel_type_sub')),df_preprocessed[i]==hotel_type_idx.hotel_type_sub,'left')
        df_preprocessed = df_preprocessed.drop(i)
        df_preprocessed = df_preprocessed.withColumnRenamed('hotel_type_sub',i).fillna(0,subset=[i])


df_preprocessed.repartition(300).write.parquet(OUT_PATH)
print('Preprocessed data done!')


# calculate ans save mean and standard deviation of numerical columns
numerical_columns = ['fprice', 'rating', 'rcount', 'searchtime', 'discount_per', 
                    'user_abp', 'place_popularity_score', 'hotel_btod', 'hotel_user_br',
                    'hotel_user_ctr', 'hotel_user_btod', 'avg_hotel_ctr',
                    'avg_hotel_btod',  'stayLength', 'abp_price_diff',
                    'hotel_ctr', 'hotel_br', 'hotel_dtob', 'hotel_placeid_ctr', 'hotel_placeid_br', 
                    'hotel_placeid_dtob', 'user_cat_ctr', 'user_cat_br',
                    'user_cat_dtob', 'distance', 'hotel_clicked_rate',
                    'cat_clicked_rate',
                    'fprice_ratio', 'rating_ratio', 'discount_per_ratio',
                    'clicked_fprice_list_10', 'clicked_rating_list_10','clicked_distance_list_10',
                    'clicked_distance_list_10_std',
                    'clicked_fprice_list_10_std', 'clicked_rating_list_10_std']




df_preprocessed.select(*numerical_columns).groupby().agg(*[F.mean(i).alias(i) for i in numerical_columns]).toPandas().to_csv('means.csv',index=False)
df_preprocessed.select(*numerical_columns).groupby().agg(*[F.stddev(i).alias(i) for i in numerical_columns]).toPandas().to_csv('sttdev.csv',index=False)


# save hotel, type and city ids for modeling
hotel_idx.select('hotelid_sub').toPandas().to_csv('hotel_idx.csv',index=False)
hotel_type_idx.select('hotel_type_sub').toPandas().to_csv('hotel_type_idx.csv',index=False)
cityid_idx.toPandas().to_csv('cityid_idx.csv',index=False)

print('Done!')
