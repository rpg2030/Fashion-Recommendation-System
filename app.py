import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D,Layer
from tensorflow.keras.applications.efficientnet import EfficientNetB3,preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

class L2NormLayer(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

model = EfficientNetB3(weights='imagenet',include_top=False,input_shape=(300,300,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
    L2NormLayer()
])

model.save('model.keras')

# Define augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# print(model.summary())

def extract_feature(img_path,model):
    
    ## Load the image and convert into array
    img = image.load_img(img_path,target_size=(300,300))
    img_array = image.img_to_array(img)
    
    
    '''
    now keras is work on batch of images , it not work on single image.
    if we have single image still we have to convert into batch of image.
    for that we use expand_dims
    '''
    expand_img_array = np.expand_dims(img_array,axis=0)
    
    
    
    
    augmented_iter = datagen.flow(expand_img_array, batch_size=1)
    augmented_img = next(augmented_iter)
    
    
    ##convert the image into proper formate
    '''
    Since EfficientNet is trained on imagenet dataset
    so we have to preprocess the image in same way as it is preprocessed in imagenet
    
    here image are converted from RGB to BGR,then zero-centered with respect to the ImageNet dataset
    '''
    preprocessed_input = preprocess_input(augmented_img)
    
    ## send the preprocessed image to model
    ## flatten the result it means convert into 1D
    result = model.predict(preprocessed_input).flatten()
    
    ## normalize the result so that all values are between 0 to 1
    normalized_result = result/norm(result)         
    
    return normalized_result


## get the path of all images
filenames = [os.path.join('images',file) for file in os.listdir('images')]


## extract the features from all images
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_feature(file,model))


## save the features and filenames
with open('features.pkl','wb') as file:
    pickle.dump(feature_list,file)

with open('filename.pkl','wb') as file:
    pickle.dump(filenames,file)

print(feature_list)