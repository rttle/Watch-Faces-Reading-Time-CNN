#libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow import data as tf_data
from keras import layers
from PIL import Image
import io

def convert_image(bytes,size):
    image=Image.open(io.BytesIO(bytes)).convert('RGB')
    image=image.resize(size, Image.Resampling.LANCZOS)
    return np.array(image)

def image_preprocess(dataframe,target,standardize,img_size):
    #changing column type (object to datetime)
    dataframe['text']=pd.to_datetime(dataframe['text'],format='%H:%M')

    #splitting into two columns
    dataframe['Hour']=dataframe['text'].dt.hour+dataframe['text'].dt.minute / 60.0 #decimal will hopefully help give more information to model
    dataframe['Hour (categorical)']=dataframe['text'].dt.hour #hour column for categorical approach
    dataframe['Minute']=dataframe['text'].dt.minute

    #target column option 2: minutes from midnight
    dataframe['Minutes from Midnight']=(dataframe['Hour (categorical)']*60)+dataframe['Minute']

    #normalizing all numerical columns
    if standardize==False and target!='categorical':
        hournorm=12+59/60
        dataframe['Hour']=dataframe['Hour']/hournorm
        dataframe['Minute']=dataframe['Minute']/60
        dataframe['Minutes from Midnight']=dataframe['Minutes from Midnight']/779

    #standardization instead of normalization
    if standardize==True and target!='categorical':
        hmean=dataframe['Hour'].mean()
        hstd=dataframe['Hour'].std()
        mmean=dataframe['Minute'].mean()
        mstd=dataframe['Minute'].std()
        mfmean=dataframe['Minutes from Midnight'].mean()
        mfstd=dataframe['Minutes from Midnight'].std()
        dataframe['Hour']=(dataframe['Hour']-hmean)/hstd
        dataframe['Minute']=(dataframe['Minute']-mmean)/mstd
        dataframe['Minutes from Midnight']=(dataframe['Minutes from Midnight']-mfmean)/mfstd
        targetstats={
            'Hour':{'mean':hmean,'std':hstd},
            'Minute':{'mean':mmean,'std':mstd},
            'Minutes from Midnight':{'mean':mfmean,'std':mfstd}
        }

    #Changing dictionary to just keep values
    dataframe['image']=dataframe['image'].apply(lambda x: x['bytes'])

    #convert byte to image
    dataframe['image']=dataframe['image'].apply(lambda x: convert_image(bytes=x,size=img_size))

    #convert to tensor
    #dataframe['image']=dataframe['image'].apply(lambda x: tf.convert_to_tensor(x, dtype=tf.float32)/255) #conversion + normalize
    dataframe['image']=dataframe['image'].apply(lambda x: keras.applications.efficientnet.preprocess_input(tf.convert_to_tensor(x.astype(np.float32)))) # conversion for use with EfficientNetB0

    if standardize==True and target!='categorical':
        return dataframe,targetstats
    else:
        return dataframe

#data augmentation setup
data_augmentation_layers = keras.Sequential([
    layers.RandomContrast(0.2),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.15,0.15)
])

#def data_augmentation(images):
#    for layer in data_augmentation_layers:
#        images = layer(images)
#    return images

def image_loader(file_name,target,batches,augmentation,img_size,standardize=False):
    df=pd.read_parquet(file_name)
    if standardize==True:
        df,targetstats=image_preprocess(df,target,standardize,img_size)
    else:
        df=image_preprocess(df,target,standardize,img_size)

    #labels
    images=list(df['image'])
    if target=='minutes':
        labels=df['Minutes from Midnight'].astype('float32').values
    elif target=='list':
        labels=df[['Hour','Minute']].astype('float32').values
    elif target=='categorical':
        labels=df[['Hour (categorical)','Minute']]

    #changing pandas dataframe to tf dataset; taking subset of dataset
    df_sub=df[:1000]
    images=list(df_sub['image'])
    labels=labels[:1000].astype('float32')
    tf_dataset=tf.data.Dataset.from_tensor_slices((images,labels))

    #taking small subset for model + splitting into training and validation
    train_ds=tf_dataset.take(800)
    val_ds=tf_dataset.skip(800)

    #dataset; data augmentation or not
    if augmentation==True:
        train_ds = train_ds.map(
            lambda img, label: (data_augmentation_layers(img), label),
            num_parallel_calls=tf_data.AUTOTUNE,
        )

    train_ds=train_ds.shuffle(800).batch(batches).prefetch(tf.data.AUTOTUNE)
    val_ds=val_ds.batch(batches).prefetch(tf.data.AUTOTUNE)

    if standardize==True:
        return train_ds, val_ds, targetstats
    else:
        return train_ds, val_ds
