# ______________________imports________________________________

import gc
import os
import time
import copy
import pickle
import datetime
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from datetime import date, timedelta
from pyspark.sql import SparkSession
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# ___________________refresh tables_____________________________

spark.sql('REFRESH TABLE ranking_service.ranking_request_log_parq')
spark.sql('REFRESH TABLE ranking_service.ranking_response_log_parq')
spark.sql('REFRESH TABLE default.analyticdeviceeventsnew')
spark.sql('REFRESH TABLE supply_onboarding_service.property')
spark.sql('REFRESH TABLE search_service.search_response_log_parq')
spark.sql('REFRESH TABLE ingestiondb.bookings_base')
spark.sql('REFRESH TABLE ranking_userevents_service.User_events_response_log')

# __________________set global variables________________________

# start date of the datapull
START_DATE = 2019110100 #2019090100 

# end date of the datapull, the duration of data is 3 months. 
# The eventual idea is to make this time independant and rely on last X clicks
END_DATE = 2019110200 #2019113024 : 1 day data to test the code

# Country = India
COUNTRY = "'1'"

# filename to save the final dataset 
FILENAME = 's3://prod-search-ranking-ml/experiments/wnd/India_POC/datapull_test'


# ________________________run queries____________________________  

# pull request infromation from ranking_service.ranking_request_log_parq (primary key : 'requestid','userid')
ranking_request_query = """
        SELECT requestid,
               userid,
               checkin,
               checkout,
               platform,
               placeid,
               cityid,
               countryid,
               nearbysearch,
               searchedlocation,
               userlocation,
               inserted_at
          FROM ranking_service.ranking_request_log_parq
          WHERE inserted_at >= '{startDate}' and inserted_at < '{endDate}'
          AND countryid in ({loc})
""".format(startDate = START_DATE, endDate = END_DATE, loc = COUNTRY)
ranking_request = spark.sql(ranking_request_query).dropna(subset=['requestid','userid']).drop_duplicates().repartition(800,'requestid')

# pull ranking response information from ranking_service.ranking_response_log_parq (primary key : 'requestid','hotelid')
# the response has been filtered to requests that were sorted by popularity (sorted by algorithm)
ranking_response_query = """
              SELECT requestid, 
                     hotelid, 
                     hotelslasherprice, 
                     hotelrank as rank, 
                     raw_hotel_type as hotel_type, 
                     raw_user_wtp,
                     user_abp,
                     raw_recommended_hotel_wc as recommended_hotel,
                     raw_mg_score as mg_score, 
                     raw_oyo_share as oyo_share
                FROM ranking_service.ranking_response_log_parq 
               WHERE sortby = 'popularity' 
                 AND inserted_at>='{startDate}' AND inserted_at<'{endDate}' 
""".format(startDate = START_DATE, endDate = END_DATE, loc = COUNTRY)
ranking_response = spark.sql(ranking_response_query).dropna(subset=['requestid','hotelid']).drop_duplicates().repartition(800,'requestid')

# Pull request ids that had atleast one click from default.analyticdeviceeventsnew (primary key : 'requestid')
hotel_details_only_query = """
             SELECT distinct requestid, dmodel 
               FROM default.analyticdeviceeventsnew 
              WHERE inserted_at>='{startDate}' and inserted_at<'{endDate}'
                AND event='hotel_details_page'        
""".format(startDate = START_DATE, endDate = END_DATE)
hotel_details_only = spark.sql(hotel_details_only_query).drop_duplicates(subset=['requestid']).repartition(800,'requestid')

# pull booking and hotel_details flag from default.analyticdeviceeventsnew (primary key : 'hotelid', 'requestid')
book_details_flag_query = """
                 SELECT hotelid,
                        requestid,
                        bookingid,
                        event
                   FROM default.analyticdeviceeventsnew
                  WHERE event in ('booking','hotel_details_page')
                    AND inserted_at >= '{startDate}' and inserted_at < '{endDate}' 
                    AND requestid!='-' 
""".format(startDate = START_DATE, endDate = END_DATE)
book_details_flag= spark.sql(book_details_flag_query).drop_duplicates().repartition(800,'requestid')

# pull search response data from search_service.search_response_log_parq (primary key : 'requestid', 'hotelid')
search_repsonse_query = """
                SELECT requestid,
                       hotelid,
                       distance,
                       fprice,
                       rating,
                       (case when distance is null then 1 else 0 end) as iscitysearch,
                       cast(ratingcount as double) as rcount,
                       popularLocationDistance,
                       popularLocationId,
                       plPopularity,
                       plConversion,
                       plHotelCount,
                       timestamp as searchtime,
                       cbreakfast as breakfastInc,
                       wizardtype,
                       dispCategory 
                  FROM search_service.search_response_log_parq
                 WHERE inserted_at >= '{startDate}' and inserted_at<'{endDate}'
""".format(startDate = START_DATE, endDate = END_DATE, loc = COUNTRY)
search_repsonse = spark.sql(search_repsonse_query).drop_duplicates().repartition(800,'requestid')

# pull realised flg for bookings from ingestiondb.bookings_base
start_date = '' + str(START_DATE)[0:4] + '-' + str(START_DATE)[4:6] + '-' + str(START_DATE)[6:8]
end_date = '' + str(END_DATE)[0:4] + '-' + str(END_DATE)[4:6] + '-' + str(END_DATE)[6:8]
realised_booking_query = """
                 SELECT id as bookingid
                   FROM ingestiondb.bookings_base
                  WHERE status in (1,2)
                    AND to_date(created_at) >= '{startDate}' and created_at < '{endDate}'
""".format(startDate = start_date, endDate = end_date)
realised_booking = spark.sql(realised_booking_query)

# get hotel level booking count, impression count and details page count from default.analyticdeviceeventsnew
event_count_query = """
            SELECT hotelid, 
                   event,
                   count(event) as eventCount 
              FROM default.analyticdeviceeventsnew
             WHERE inserted_at >= '{startDate}' and inserted_at < '{endDate}'
               AND event in ('hotel_details_page','booking','hotel_impression')
               AND  requestid!='-' 
          GROUP BY hotelid, event
""".format(startDate = START_DATE, endDate = END_DATE)
event_count = spark.sql(event_count_query).drop_duplicates()

event_count = event_count.groupby('hotelid').\
                          pivot('event', ['hotel_details_page','booking','hotel_impression']).\
                          sum('eventCount').\
                          select(F.col('hotelid'),
                                 F.col('hotel_details_page').alias('details_count'),
                                 F.col('booking').alias('booking_count'),
                                 F.col('hotel_impression').alias('impression_count'))

# get user level booking count, impression count and details page count etc. from default.analyticdeviceeventsnew
user_events_query = """
            SELECT hotelid,
                   uuid as requestid,
                   impressioncount,
                   detailcount,
                   bookingcount,
                   userimpressioncount,
                   userdetailcount,
                   userbookingcount
              FROM ranking_userevents_service.User_events_response_log 
             WHERE inserted_at>='{startDate}' 
               AND inserted_at<'{endDate}' 
               AND hotelid!='-' 
               AND uuid!='-' 
""".format(startDate = START_DATE, endDate = END_DATE)
user_event_count = spark.sql(user_events_query).drop_duplicates().repartition(800,'requestid')

# _____________________________merge the tables ________________________________

ranking_request = ranking_request.join(hotel_details_only,on='requestid',how='inner').repartition(800,'requestid')
ranking_hcat_df = ranking_request.join(ranking_response,on='requestid',how='inner').repartition(800,'requestid')
ranking_booking_df = ranking_hcat_df.join(book_details_flag,on=['requestid','hotelid'],how='left').repartition(800,'requestid')
ranking_search_df = ranking_booking_df.join(search_repsonse,on=['requestid','hotelid'],how='left').repartition(800,'requestid')
features_data_df = ranking_search_df.join(realised_booking.select('bookingid',F.lit(1).alias('realised_flag')),on='bookingid',how='left').repartition(800,'requestid')
features_df = features_data_df.join(event_count,on='hotelid',how='left').repartition(800,'requestid')
features_df = features_df.join(user_event_count,on=['requestid','hotelid'],how='left').repartition(800,'requestid')
features_df = features_df.where('(fprice is not null) and (hotelslasherprice is not null)')

# ________________________ write the dataset to S3 _____________________________

features_df.repartition(800,'userid').write.parquet(FILENAME)

print("""Data pull complete""")
