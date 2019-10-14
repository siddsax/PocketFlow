''' Image dataset processing (IDP) for machine learning '''


import io
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import os 
import random
from os.path import isfile, join
__author__ = "Andres Vourakis"
__email__ = "andresvourakis@gmail.com"
__license__ = "GPL"
__data__ = "May 25, 2017"


def image_to_byte_array(image, class_index, size):

    img = Image.open(image)
    
    #Resize
    img = img.resize(size) #TODO Check if resizing to given dim can be done

    #Convert image to 3 dimensional array
    img_array = np.array(img)

    #Convert 3 dimensional array into row major order
    img_array_R = img_array[:,:,0].flatten()
    img_array_G = img_array[:,:,1].flatten()
    img_array_B = img_array[:,:,2].flatten()
    class_index = [class_index]

    # Turn row-major array into bytes
    #img_byte_array = np.concatenate((img_array_R, img_array_G, img_array_B)).tobytes() #Turn into row-major byte array
    img_byte_array = np.array(list(class_index) + list(img_array_R) + list(img_array_G) + list(img_array_B), np.uint8) #Turn into row-major byte array
    
    return img_byte_array

def create_meta_data(class_labels, destination):
    
    '''
        TODO: Check if directory exists
    '''
    
    file_name = 'batches_meta.txt'
    file_path = os.path.join(destination, file_name)
    with open(file_path, 'w') as file:
        for label in class_labels:
            file.write(str(label) + '\n')


def label_to_index(class_labels, class_label):
    return class_labels.index(class_label)

def open_batch(destination, CURRENT_BATCH, test=False):
    if test:
        file_name = 'test_batch_' + str(CURRENT_BATCH) + '.bin'
    else:
        file_name = 'data_batch_' + str(CURRENT_BATCH) + '.bin'
    file_path = os.path.join(destination, file_name)
    return open(file_path, 'wb')

def close_batch(file):
    file.close()

def process_dataset(source, destination, size=(32, 32), batch_size=1):
    """ 
        Processes dataset into binary version of CIFAR-10 dataset

    Args:
        source: Abosulute path to directory containing subdirectories of image datasets.
        destination: Absolute path of directory where to save process image datasets.
        size (default = (32,32)): square dimensions (width and height) to resize images
        batch (default = 1): Number of batches to divide image dataset.
        
    """

    df = pd.read_csv(os.path.join(source, 'labels.csv'), header='infer')
    class_labels = list(df)[1:]
    train_len = len(df) // 2
    valid_len = len(df) * 3 // 4


    create_meta_data(class_labels, destination) #TODO: Check time complex. 

    # class_labels = next(os.walk(source))[1]
    #dataset_size = len(next(os.walk(source))[2]) #Only gives tot number of files in current directory

    # for phrase in ['train', 'val', 'test']:

    #     dataset_size = dataSize[phrase]
    
    REACHED_BATCH_MAX = False
    CURRENT_BATCH = 1

    #create meta data file
    batch = open_batch(destination, CURRENT_BATCH)
    root = source + 'images'
    files = [f for f in os.listdir(root) if isfile(join(root, f))]
    l2 = [int(i.split('.')[0]) for i in files]
    l3 = np.argsort(l2)
    files = [files[i] for i in l3]
    # import pdb;pdb.set_trace()

    index = list(range(len(files)))
    random.shuffle(index)
    files = [files[i] for i in index]

    #load data and output data
    fg = 1
    for i, j in enumerate(index):
        fl = files[i]
        class_index = np.argmax([df[k][j] for k in class_labels])
        file_path = os.path.join(root, fl)

        if(i % batch_size == 0 and i !=0):

            close_batch(batch)
            REACHED_BATCH_MAX = True
            print(i, file_path, class_labels[class_index])

        # print(class_index, i)
            
        if(REACHED_BATCH_MAX):
            CURRENT_BATCH += 1
            if i > valid_len:
                if fg:
                    CURRENT_BATCH = 0
                    fg = 0
                batch = open_batch(destination, CURRENT_BATCH, test=True)                
            else:
                batch = open_batch(destination, CURRENT_BATCH)
            REACHED_BATCH_MAX = False
        
        image_byte_array = image_to_byte_array(file_path, class_index, size)
        
        batch.write(image_byte_array)
        # import pdb; pdb.set_trace()
        

    close_batch(batch) 

