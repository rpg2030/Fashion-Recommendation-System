import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.efficientnet import EfficientNetB3,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import cv2

from sklearn.neighbors import NearestNeighbors

feature_list = np.array(pickle.load(open('features.pkl','rb')))
filename = pickle.load(open('filename.pkl','rb'))

model = EfficientNetB3(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

model.summary()

img = image.load_img('sample/shoes.jpeg',target_size=(224,224))
img_array = image.img_to_array(img)
expand_img_array = np.expand_dims(img_array,axis=0)
preprocessed_input = preprocess_input(expand_img_array)
result = model.predict(preprocessed_input).flatten()
normalized_result = result/norm(result)

neighbour = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbour.fit(feature_list)

distance,indices = neighbour.kneighbors([normalized_result])

print(indices)
print(distance)

for file in indices[0]:
    temp_size = cv2.imread(filename[file])
    cv2.imshow('output',cv2.resize(temp_size,(512,512)))
    cv2.waitKey(0)