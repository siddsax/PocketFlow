from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import os 
from dataset_to_cifar import process_dataset


def mimlDataset(dataset_directory, isTrain=True):
#     # dataset_directory = './miml_dataset'
    df = pd.read_csv(os.path.join(dataset_directory, 'labels.csv'), header='infer')
#     show_n_records = 3 #@param {type:"integer"}
#     # We don't use 'blury' for downsapled images :-), and bright because of lack of data
#     if 'card' in dataset_directory:
#         df.drop(columns=['blurry', 'bright'], inplace=True)
#     print(df[:show_n_records])
#     print(df.columns)

#     labels = list(df)[1:]
#     print(labels)
#     filenames = list(df)[0]
#     print("Filenames column name:", filenames)
#     labels_txt = '\n'.join(labels)

#     with open('labels.txt', 'w') as f:
#         f.write(labels_txt)

#     horizontal_flip = True #@param {type:"boolean"}
#     vertical_flip = False#@param {type:"boolean"}
#     # TODO add augmentation params
#     datagen_kw = dict(rescale=1./255, 
#                     horizontal_flip=horizontal_flip,
#                     vertical_flip=vertical_flip)
#     datagen = tf.keras.preprocessing.image.ImageDataGenerator(datagen_kw)
#     test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
#     image_wh = 224
#     target_size = (image_wh, image_wh)
#     train_len = len(df) // 2
#     valid_len = len(df) * 3 // 4
#     seed = 1
#     batch_size = 32 #@param {type:"integer"}
#     train_generator = datagen.flow_from_dataframe(
#         dataframe=df[:train_len],
#         directory=dataset_directory,
#         x_col=filenames,
#         y_col=labels,
#         batch_size=batch_size,
#         seed=seed,
#         shuffle=True,
#         class_mode="other",
#         target_size=target_size)
#     valid_generator = test_datagen.flow_from_dataframe(
#         dataframe=df[train_len:valid_len],
#         directory=dataset_directory,
#         x_col=filenames,
#         y_col=labels,
#         batch_size=batch_size,
#         seed=seed,
#         shuffle=True,
#         class_mode="other",
#         target_size=target_size)
#     test_generator = test_datagen.flow_from_dataframe(
#         dataframe=df[valid_len:],
#         directory=dataset_directory,
#         x_col=filenames,
#         y_col=labels,
#         batch_size=1,
#         seed=seed,
#         shuffle=False,
#         class_mode="other",
#         target_size=target_size,
#         horizontal_flip=True)


#     # image_batch, label_batch = next(train_generator)
#     # print("Image batch shape: ", image_batch.shape)
#     # print("Label batch shape: ", label_batch.shape)

#     if isTrain:
#         return train_generator, valid_generator
#     else:
#         return test_generator


# Set square dimensions of images
size = (224,224) # 32 by 32 pixels
batch_size = 32

# Set number of batches

# Source of image dataset (Use absolute path)
source = '../miml_dataset/'
target = source + 'bin/'
if not os.path.exists(target):
    os.makedirs(target)

# Process dataset
process_dataset(source, target, size, batch_size = batch_size)
