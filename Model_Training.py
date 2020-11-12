# -*- coding: utf-8 -*-


from zipfile import ZipFile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, datasets, models
import pandas as pd
import numpy as np
import cv2

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

tranch1 = pd.read_csv('tranch1_labels.csv')
tranch2 = pd.read_csv('tranch2_labels.csv')
tranch3 = pd.read_csv('tranch3_labels.csv')

for i in range(len(tranch1)):
  tranch1.file_name[i] = 'tranch1/' + tranch1.file_name[i]
for i in range(len(tranch2)):
  tranch2.final_url[i] = 'tranch2/' + tranch2.final_url[i]
for i in range(len(tranch3)):
  tranch3.final_url[i] = 'tranch3/' + tranch3.final_url[i]


tranch2['file_name'] = tranch2.final_url
tranch3['file_name'] = tranch3.final_url

all_data = pd.concat([tranch1,tranch2,tranch3])
total_num = len(all_data)
all_data = pd.concat([all_data, all_data])
all_data.index = range(len(all_data))
all_data.drop(['final_url', 'exception_case'], axis = 1, inplace = True)

for i in range(total_num, len(all_data)):
  tranch = all_data.file_name[i].split('/')[0]
  filename = all_data.file_name[i].split('/')[1]
  all_data.file_name[i] = tranch + '_edge/' + filename

all_data.drop(all_data[all_data.primary_posture == 'Unknown'].index, axis = 0, inplace = True)
all_data.drop(all_data[all_data.primary_posture.isnull()].index, axis = 0, inplace = True)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size = 128
epochs = 15
IMG_HEIGHT = 299
IMG_WIDTH = 299
image_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
train_data_gen = image_gen.flow_from_dataframe(all_data, x_col = 'file_name', y_col = 'primary_posture',
                         batch_size = batch_size, shuffle = True, target_size = (IMG_HEIGHT, IMG_WIDTH), 
                         class_mode = 'categorical', subset = 'training')

validation_data_gen = image_gen.flow_from_dataframe(all_data, x_col = 'file_name', y_col = 'primary_posture',
                         batch_size = batch_size, shuffle = True, target_size = (IMG_HEIGHT, IMG_WIDTH), 
                         class_mode = 'categorical', subset = 'validation')


#MobileNet pretrained on imagenet
'''
from tensorflow.keras.applications import EfficientNetB7, MobileNet
base_model=MobileNet(weights='imagenet',input_shape = (224,224,3),include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
base_model.trainable = True
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
'''


#Inception-V3 pretrained on imagenet

from tensorflow.keras.applications import InceptionResNetV2, InceptionV3
from tensorflow.keras.models import load_model 
base_model=InceptionV3(weights='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',input_shape = (299,299,3),include_top=False)
base_model.trainable = False

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


hists = []
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model.fit_generator(train_data_gen, steps_per_epoch = 28279//batch_size, epochs=100, validation_data = validation_data_gen, 
                    validation_steps = 7069//batch_size, callbacks=callbacks, verbose = 2)
hists.append(model.history.history)

model.save('InceptionV3-Unfrozen-Edged.h5')

acc = []
val_acc = []
for i in range(len(hists)):
    acc += hists[i]["accuracy"]
    val_acc += hists[i]["val_accuracy"]
hist_df = pd.DataFrame({"# Epoch": [e for e in range(1,len(acc)+1)],"Accuracy": acc, "Val_accuracy": val_acc})
hist_df.plot(x = "# Epoch", y = ["Accuracy","Val_accuracy"])
plt.title("Accuracy vs Validation Accuracy")
plt.savefig('Accuracy.jpg')
plt.show()