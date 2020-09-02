import re
import gc
import os
import joblib
import pandas
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
pd.set_option('max_columns',None)
import logging

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import  f1_score, classification_report
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask import send_file
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow import feature_column
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Dense, DenseFeatures, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate, Embedding, Reshape, Average, Subtract, Dot
from tensorflow.keras.layers import Flatten, Lambda, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
dt_stamp = datetime.strftime(datetime.now(),'%Y-%m-%d_%H-%M-%S')

#columns
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

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


embed_features = ['hotelid','hotel_type','cityid']

embed_sequence = ['clicked_hotelid_list_10']

categorical_columns = ['platform', 'is_weekend_checkin',
                       'is_weekend_searchDate',  'is_city_search']

distance_bucket = ['distance_buckets']

target = ['target_flag']

clicked_hotelid_columns=['clicked_hotelid_{}'.format(i+1)for i in range(10)]
clicked_hotel_type_columns=['clicked_hotel_type_{}'.format(i+1)for i in range(10)]


# load the entire dataset
#directory = '/home/madhur.popli/experiments/wnd/India_POC/0304_data_processed_with_preprep'
directory = '/home/prabhjot.singh/wnd_pretrained_4/test'
mean_data = 'means.csv'
std_data = 'sttdev.csv'
hotel_idx = 'hotel_idx.csv'
cityid_idx = 'cityid_idx.csv'
hotel_type_idx = 'hotel_type_idx.csv'

mean_df = pd.read_csv(mean_data)
std_df = pd.read_csv(std_data)


hotel_idx = pd.read_csv(hotel_idx)
cityid_idx = pd.read_csv(cityid_idx)
hotel_type_idx = pd.read_csv(hotel_type_idx)

unique_hotelid_raw = [str(i) for i in [0] + list(hotel_idx['hotelid_sub'])]
unique_cityid_raw = [str(i) for i in [0] + list(cityid_idx['cityid_sub'])]
unique_hotel_type_raw = list(hotel_type_idx['hotel_type_sub'].astype(int))


unique_hotelid_dict = dict(zip(unique_hotelid_raw,range(len(unique_hotelid_raw))))
unique_hotel_type_dict = dict(zip(unique_hotel_type_raw,range(len(unique_hotel_type_raw))))
unique_cityid_dict = dict(zip(unique_cityid_raw,range(len(unique_cityid_raw))))

unique_hotelid = list(unique_hotelid_dict.values())
unique_hotel_type = list(unique_hotel_type_dict.values())
unique_cityid = list(unique_cityid_dict.values())

class DataGenerator(Sequence):

        def __init__(self, directory, batch_size):
            self.directory = directory
            self.batch_size = batch_size
            self.numeric_columns = None
            self.generator = self.data_generator()

        def data_generator(self):
            user_i = 1
            for file_name in os.listdir(self.directory):
                if file_name != '_SUCCESS':
                    user_i +=1
                    print('loop : {}'.format(user_i))
                    # read_data
                    df = pd.read_parquet(directory + '/' + file_name)
                    #df = df[df['cityid'].isin(['8','2','4','1','3','14','12','11','5','87','7','24','43','111','27','88','17','33','21','128','28','10','9','30','65','63','59'])]
                    #df = df[df['inserted_at'].astype(int) >= 2019112100]
                    df[numerical_columns] = df[numerical_columns].astype(np.float)
                    df[['hotel_type','distance_buckets']] = df[['hotel_type','distance_buckets']].astype(int)

                    df['hotelid'] = df['hotelid'].map(unique_hotelid_dict)
                    df['hotel_type'] = df['hotel_type'].map(unique_hotel_type_dict)
                    df['cityid'] = df['cityid'].map(unique_cityid_dict)

                    for i in clicked_hotelid_columns:
                        df[i] = df[i].map(unique_hotelid_dict)
                    for i in clicked_hotel_type_columns:
                        df[i] = df[i].map(unique_hotel_type_dict)

                    # define target flag
                    df['target_flag'] = df['target'].map({'realised':'booking',
                                            'hotel_details_page':None,
                                            'booking':'booking',
                                            None:None})

                    df_out = df.copy(deep=True)
                    #print(df_out.head())
                    # select subset of columns
                    df = df[sorted(set(numerical_columns +
                            categorical_columns + distance_bucket + target +
                            clicked_hotelid_columns +
                            clicked_hotel_type_columns + ['hotelid','hotel_type','cityid']))]

                    # apply scaling
                    scaled_df = df[numerical_columns].sub(list(mean_df[numerical_columns].iloc[0]))
                    scaled_df = scaled_df.div(list(mean_df[numerical_columns].iloc[0]))

                    df[numerical_columns] = scaled_df

                    # categorical columns
                    df = df.fillna(0)

                    X = df.drop(columns=target)
                    X['android_app'] = np.where(X['platform'] == 'android_app',1,0)
                    X['website'] = np.where(X['platform'] == 'website',1,0)
                    X['ios_app'] = np.where(X['platform'] == 'ios_app',1,0)
                    X.drop(columns=['platform'],inplace=True)

                    y_ = df[target].fillna('no_action')
                    y = pd.get_dummies(y_)
                    y = np.argmax(y.values,axis=1)

                    numeric_columns = [i for i in X.columns if i not in  target +
                                    clicked_hotelid_columns +
                                    clicked_hotel_type_columns + ['hotelid','hotel_type','cityid'] + distance_bucket]

                    self.numeric_columns = numeric_columns

                    X_out = [X[self.numeric_columns],
                    X['hotelid'].astype(int),
                    X['hotel_type'].astype(int),
                    X[clicked_hotelid_columns].astype(int),X['cityid'].astype(int)] + [X['clicked_hotel_type_{}'.format(i+1)].astype(int) for i in range(10)]  + [X['distance_buckets'],X['is_city_search']]
                    X_out = [np.asarray(i) for i in X_out]
                    y_out = np.asarray(y)

                    yield (X_out,y_out,df_out)



model = keras.models.load_model('wnd_2250.h5')

generator_class = DataGenerator(directory,1024)
generator = generator_class.data_generator()

uploads_dir = '/home/prabhjot.singh/wnd_pretrained_4/test'

import timeit

@app.route("/predictHotelsRankForRequest", methods=['POST','PUT'])
def predictRank():
    file = request.files['file']
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    
    for X_train,y_train,df_out in generator:
        start = timeit.default_timer()
        #print(y_train.shape)
        pred_output = model.predict(X_train)
        df_out['predicted_val'] = pred_output
        df_out['new_rank'] = df_out.groupby('requestid')['predicted_val'].rank(method='first',ascending=False).apply(int)
        df_out['current_rank'] = df_out.groupby('requestid')['rank'].rank(method='first').apply(int)

        df_out.to_csv('validation_2.csv',mode = 'w',index=False)
        #print(df_out[['requestid','hotelid','userId','current_rank','new_rank']])
        stop = timeit.default_timer()
        #print('Time: ', stop - start)
        app.logger.error('Time taken: '+ str(stop - start))

    return send_file('validation_2.csv', attachment_filename='validation_2.csv')


if __name__=="__main__":
    app.run(host="10.15.10.211",port=5000, debug=True)
