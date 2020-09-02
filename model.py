# _______________________________imports________________________________________
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

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import  f1_score, classification_report

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


# __________________________________ model inputs ___________________________________________

# identify the numerical columns in the input features
NUMERICAL_COLUMNS = ['fprice', 'rating', 'rcount', 'searchtime', 'discount_per', 
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

# identify the categorical columns in the input features (one hot encoded variables)
CATEGORICAL_COLUMNS = ['platform', 'is_weekend_checkin', 
                       'is_weekend_searchDate',  'is_city_search']

DISTANCE_BUCKET = ['distance_buckets']
TARGET = ['target_flag']
CLICKED_HOTELID_COLUMNS=['clicked_hotelid_{}'.format(i+1)for i in range(10)]
CLICKED_HOTEL_TYPE_COLUMNS=['clicked_hotel_type_{}'.format(i+1)for i in range(10)]

DIRECTORY = '/home/madhur.popli/experiments/wnd/India_POC/0304_data_processed_with_preprep_train' # train data path
MEAN_DATA = 'means.csv' # numerical data mean
STD_DATA = 'sttdev.csv' # numerical data stddev
HOTEL_IDX = 'hotel_idx.csv' 
CITYID_IDX = 'cityid_idx.csv'
HOTEL_TYPE_IDX = 'hotel_type_idx.csv'

INITIAL_EMBEDDINGS = '1_embedded.csv'

# weights to remove class imbalance
WEIGHTS_CLASS = {0:1,1:175}

# _____________________________ load datasets _______________________________________

mean_df = pd.read_csv(MEAN_DATA)
std_df = pd.read_csv(STD_DATA)
hotel_idx = pd.read_csv(HOTEL_IDX)
cityid_idx = pd.read_csv(CITYID_IDX)
hotel_type_idx = pd.read_csv(HOTEL_TYPE_IDX)
embedding_df = pd.read_csv(INITIAL_EMBEDDINGS,names =['property_id','latitude','longitude','oyo_id','country_name','country_id','city_name','city_id','embedding'] )

# _______________________preprocessing and transformations ___________________________

# read hotelids, type, cityid 
# unique_hotelid_raw = [str(i) for i in [0] + list(hotel_idx['hotelid_sub'])]
unique_hotel_type_raw = list(hotel_type_idx['hotel_type_sub'].astype(int))
unique_cityid_raw = [str(i) for i in [0] + list(cityid_idx['cityid_sub'])]

# create dictionary to label encode hotel, type and city
# unique_hotelid_dict = dict(zip(unique_hotelid_raw,range(len(unique_hotelid_raw))))
unique_hotel_type_dict = dict(zip(unique_hotel_type_raw,range(len(unique_hotel_type_raw))))
unique_cityid_dict = dict(zip(unique_cityid_raw,range(len(unique_cityid_raw))))

# encoded hotel, type, city
# unique_hotelid = list(unique_hotelid_dict.values())
unique_hotel_type = list(unique_hotel_type_dict.values())
unique_cityid = list(unique_cityid_dict.values())

# preprocess embeddigns (to be changed if the source of embeddings change)
def get_list(x):
    x = x[1:-1].replace('\n',' ').split(' ')
    x = [float(i) for i in x if i != '']
    return x

hotel_emb = embedding_df['embedding'].apply(get_list)
hotel_emb = np.stack(hotel_emb)
hotel_emb = np.concatenate((hotel_emb.mean(axis=0).reshape([1,32]),hotel_emb))

hotel_lkp = embedding_df['property_id'].apply(str).reset_index()
hotel_lkp['index'] += 1
hotel_lkp = hotel_lkp.set_index('property_id').to_dict()['index']
hotel_lkp['0'] = 0

unique_hotelid_raw = list(hotel_lkp.keys())
unique_hotelid_dict = hotel_lkp
unique_hotelid = list(unique_hotelid_dict.values())

# save lookups for scoring
joblib.dump(unique_hotelid_dict,'unique_hotelid_dict.sav')
joblib.dump(unique_hotel_type_dict,'unique_hotel_type_dict.sav')
joblib.dump(unique_cityid_dict,'unique_cityid_dict.sav')

# data generator for model training
class DataGenerator(Sequence):
    """
    generate partitions of data for model training
    """
    def __init__(self, directory):
        self.directory = directory

    def data_generator(self):
        """
        data generator method
        """
        for file_name in os.listdir(self.directory): # read files from a folder saved during preprocessing (parquet files)
            if file_name != '_SUCCESS':
                # read_data
                df = pd.read_parquet(directory + '/' + file_name)
                df[NUMERICAL_COLUMNS] = df[NUMERICAL_COLUMNS].astype(np.float)
                df[['hotel_type','distance_buckets']] = df[['hotel_type','distance_buckets']].astype(int)
                
                # label endocode hotelid, city, type
                df['hotelid'] = df['hotelid'].map(unique_hotelid_dict)
                df['hotel_type'] = df['hotel_type'].map(unique_hotel_type_dict)
                df['cityid'] = df['cityid'].map(unique_cityid_dict)

                # label encode clickstream data
                for i in CLICKED_HOTELID_COLUMNS:
                    df[i] = df[i].map(unique_hotelid_dict)
                for i in CLICKED_HOTEL_TYPE_COLUMNS:
                    df[i] = df[i].map(unique_hotel_type_dict)

                # define target flag
                df['target_flag'] = df['target'].map({'realised':'booking',
                                        'hotel_details_page':None,
                                        'booking':'booking',
                                        None:None})

                # select subset of columns 
                df = df[sorted(set(NUMERICAL_COLUMNS + 
                        CATEGORICAL_COLUMNS + DISTANCE_BUCKET + TARGET + 
                        CLICKED_HOTELID_COLUMNS +
                        CLICKED_HOTEL_TYPE_COLUMNS + ['hotelid','hotel_type','cityid']))]

                # apply scaling to numerical columns
                scaled_df = df[NUMERICAL_COLUMNS].sub(list(mean_df[NUMERICAL_COLUMNS].iloc[0]))
                scaled_df = scaled_df.div(list(std_df[NUMERICAL_COLUMNS].iloc[0]))
                df[NUMERICAL_COLUMNS] = scaled_df

                # categorical columns
                df = df.fillna(0)
                
                # randomise data
                df = df.sample(frac=1)

                # one hot encode categorical data
                X = df.drop(columns=TARGET)
                X['android_app'] = np.where(X['platform'] == 'android_app',1,0)
                X['website'] = np.where(X['platform'] == 'website',1,0)
                X['ios_app'] = np.where(X['platform'] == 'ios_app',1,0)
                X.drop(columns=['platform'],inplace=True)

                # create target
                y_ = df[TARGET].fillna('no_action')
                y = pd.get_dummies(y_)
                y = np.argmax(y.values,axis=1)

                numeric_columns = [i for i in X.columns if i not in  TARGET +
                                CLICKED_HOTELID_COLUMNS +
                                CLICKED_HOTEL_TYPE_COLUMNS + ['hotelid','hotel_type','cityid'] + DISTANCE_BUCKET]
                
                self.numeric_columns = numeric_columns

                # create model input (list of lists) and target
                X_out = [X[self.numeric_columns],
                X['hotelid'].astype(int),
                X['hotel_type'].astype(int),
                X[CLICKED_HOTELID_COLUMNS].astype(int),X['cityid'].astype(int)] + [X['clicked_hotel_type_{}'.format(i+1)].astype(int) for i in range(10)]  + [X['distance_buckets'],X['is_city_search']]
                X_out = [np.asarray(i) for i in X_out]
                y_out = np.asarray(y)

                yield (X_out,y_out)
    

# ________________________________MODEL___________________________________

# ______wide model start ________

# distance bucket input
distance_bucket_input = Input(shape=(1,), dtype='int64', name='distance_bucket')
unique_distance_buckets = list(range(10))
distance_bucket_column = feature_column.categorical_column_with_vocabulary_list('distance_bucket_input', unique_distance_buckets)

# is city search input
is_city_search_input = Input(shape=(1,), dtype='int64', name='is_city_search')
unique_is_city_search = [0,1]
is_city_search_column = feature_column.categorical_column_with_vocabulary_list('is_city_search_input', unique_is_city_search)

# interation features between distance bucket and city search input
distance_city_cross_feature = feature_column.crossed_column([distance_bucket_column, is_city_search_column], hash_bucket_size=20,hash_key=42)
distance_city_cross_indicator_feature = feature_column.indicator_column(distance_city_cross_feature)
distance_city_cross_dense = layers.DenseFeatures(distance_city_cross_indicator_feature)({'distance_bucket_input':distance_bucket_input,
                                                                                        'is_city_search_input':is_city_search_input})

# create input for hotel type
hotel_type_input = Input(shape=(1,), dtype='int64', name='hotel_type')
hotel_type = feature_column.categorical_column_with_vocabulary_list(
         'hotel_type', unique_hotel_type)

# create hotel type click stream inputs and interaction with candidate hotel type
clicked_hotel_type_input_sparse = {}
crossed_hotel_type_dense = {}
for i in CLICKED_HOTEL_TYPE_COLUMNS:
    clicked_hotel_type_input_sparse[i] = Input(shape=(1,), dtype='int64', name=i)
    categorical_col = feature_column.categorical_column_with_vocabulary_list(
                      i, unique_hotel_type)
    crossed_feature = feature_column.crossed_column([hotel_type, categorical_col], hash_bucket_size=196,hash_key=42)
    indicator_feature = feature_column.indicator_column(crossed_feature)
    crossed_hotel_type_dense[i] = layers.DenseFeatures(indicator_feature)({'hotel_type':hotel_type_input,
                                                                        i:clicked_hotel_type_input_sparse[i]})

# pooling layer for type_clickstream interation features
hotel_type_sparse_layer = layers.Add()(crossed_hotel_type_dense.values())

# _____________ wide model end _____________

# _____________deep model start_____________

# city id imput and embedding map
cityid_input = Input(shape=(1,), dtype='int64', name='cityid')
cityid_embedding = Embedding(len(unique_cityid)+1, len(unique_cityid)+1, input_length=10, weights=[np.identity(len(unique_cityid)+1)],trainable=False)(cityid_input)
cityid_embedding = Flatten()(cityid_embedding)

# create hotel embedding and pooling layer
hotelid_embedding = Embedding(len(unique_hotelid), 32, input_length=10,weights=[hotel_emb],trainable=True)
avg_layer = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1))

# hotel id input and embedding map
hotelid_input = Input(shape=(1,), dtype='int64', name='hotelid')
hotelid_embed = hotelid_embedding(hotelid_input)
hotelid_embed_flatten = Flatten()(hotelid_embed)

# # booked hotel id input and ebedding map
# booked_hotelid_input = Input(shape=(3,), dtype='int64', name='booked_hotelids')
# booked_hotelid_embed = hotelid_embedding(booked_hotelid_input)
# booked_hotelid_avg = avg_layer(booked_hotelid_embed)

# clickstream input, embedding map and pooling
clicked_hotelid_input = Input(shape=(10,), dtype='int64', name='clicked_hotelids')
clicked_hotelid_embed = hotelid_embedding(clicked_hotelid_input)
clicked_hotelid_avg = avg_layer(clicked_hotelid_embed)

# clickstream and candidate hotel interaction layer
embedding_agg_layer = Dot(axes=1)([hotelid_embed_flatten,clicked_hotelid_avg])

# conitnuos input layer
n_var = 42
continous_inputs = Input(shape=(n_var,), dtype='float32', name='continous_variables')

# concatenate deep layer inputs
concated_deep_layer = Concatenate()([continous_inputs,
                                     embedding_agg_layer,
                                     cityid_embedding
                                    ])

# deep layers, each layer is followed by bath normalisation layer
d = Dense(64,activation='relu')(concated_deep_layer)
d = BatchNormalization()(d)
d = Dense(32,activation='relu',name='deep_1')(d)
d = BatchNormalization()(d)
d = Dense(16,activation='relu',name='deep_2')(d)
d = BatchNormalization()(d)
d = Dense(8,activation='relu',name='deep_3')(d)
d = BatchNormalization()(d)

# ______________ deep model end _______________

# concatenate wide and deep layers
wnd = Concatenate()([d,hotel_type_sparse_layer,distance_city_cross_dense])

# classification sigmoid layer
out = Dense(1, activation='sigmoid', name='sigmoid',kernel_regularizer=l2(0.01))(wnd)

# assemble all inputs
inputs = [continous_inputs,hotelid_input,hotel_type_input,clicked_hotelid_input,cityid_input] + list(clicked_hotel_type_input_sparse.values()) + [distance_bucket_input,is_city_search_input]

# create model
model = Model(inputs = inputs,
              outputs = out)

# define optimiser and hyperparams
optimizer = keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999)

# compile model
model.compile(optimizer=optimizer,loss = 'binary_crossentropy')

# __________________________ model training ___________________________

# instantiate data generator class
generator_class = DataGenerator(DIRECTORY)

# each bath has ~225 files, total number of loops required = 225 * epochs
epochs = 10
n_files = 225
for i in range(epochs*n_files):
    try:
        X_train,y_train = next(generator)
    except: # generator will end after 225 iterations, re-initilaise 
        generator = generator_class.data_generator()
        X_train,y_train = next(generator)

    # fit the model, provide the valdation dataset here
    model.fit(X_train,
              y_train,
              epochs=1,
              batch_size=1024,
              class_weight = WEIGHTS_CLASS,
              verbose=1,
              use_multiprocessing=True)
    
    # save model outputs
    if i%n_files == 0:
        model.save('wnd_{}.h5'.format(i))  
        input_ = [layer.output for layer in model.layers][0]
        output_ = [layer.output for layer in model.layers][2]
        model_new = Model(input_,output_)
        embed = model_new.predict(list(range(len(unique_hotelid))))
        joblib.dump(embed.reshape([len(unique_hotelid),32]),'hotel_embed_{}.sav'.format(i))  
	
        # predictions =np.round( model.predict(X_train))
        # print(classification_report(y_train, predictions))

model.save('wnd_{}.h5'.format(i)) 
