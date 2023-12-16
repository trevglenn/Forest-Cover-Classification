## This where we will tune hyperparameters and further edit our dl_forestcover.py model and surrounding code to help improve efficiency and performance of model

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## Step One - Acquire Data

## Import Data from csv
data = pd.read_csv("forest_cover_classification\cover_data.csv")

## Gather info on our data
#print(data.head())
#print(data.info())
## 581,012 entries, 0 null counts
## All dtypes are int64, however some such as Elevation and Slope are measurements, while others such as Wilderness_Area(n) and Soil_Type(n) are Booleans
## class is our tree cover type classification ranging from 1-7
## TBD: May need to increase weight parameters of certain booleans or measurements on model to improve accuracy


## Step Two - Preprocessing

## Separate into labels(y) and features (x)

y = data.iloc[:, -1]

x = data.iloc[:, 0:-1]
## One-Hot Label Encoding
x = pd.get_dummies(x)

## Step 3 - Splitting and Balancing

## train_test_split()
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.33, random_state=72)

## StandardScaler() scale our numeric feature columns to be properly interpreted by our model
sc = StandardScaler()

## Using fit_transform() on x_train to fit and tranform data for our model
x_train = sc.fit_transform(x_train)

## Using .transform() on x_test to transform data for our model
x_test = sc.transform(x_test)

## Calling LabelEncoder() so that we can normalize our label set(Y)
#le = LabelEncoder()

## Encoding our labels as strings to properly pass through model
#y_train = le.fit_transform(y_train.astype(str))
#y_test = le.transform(y_test.astype(str))

## Converting class vectors back to Binary Class Matrix for our model to interpret
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)


## Step 4 - designing and building our model

def design_model(num_features):

    model = Sequential()
    
    model.add(InputLayer(input_shape=(x_train.shape[1], )))
    
    #model.add(Dense(256, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.3))
    #model.add(Dense(32, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.1))
    
    model.add(Dense(8, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

## 45,927 total and trainable parameters likely too many will comment out hidden layers to reduce

## Call model on training data
cols = data.columns.tolist()
features, label = cols[:-1], cols[-1]

num_features = len(features)
model = design_model(num_features)

## Model Summary
print('Summary of Model:')
model.summary()
## Can put Early Stopping here if needed
es = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
## Fit Model
model.fit(x_train, y_train, epochs=12, batch_size=1000, verbose=1, validation_split=0.15, callbacks=[es])

## Step 5 - Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {loss}')
print(f'Test accuracy: {acc}')
# Classification Report
y_estimate = model.predict(x_test, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)

class_names = ['Spruce/Fir', 'Lodgepole Pine',
                   'Ponderosa Pine', 'Cottonwood/Willow',
                   'Aspen', 'Douglas-fir', 'Krummholz']

print(classification_report(y_test, y_estimate, target_names=class_names))

## Trainable Params: 11,832
## Epoch 11: Early Stopping
## Test loss: 0.33595380187034607
## Test accuracy: 0.8614833950996399
#                   precision    recall  f1-score   support

#       Spruce/Fir       0.86      0.87      0.86     63552
#   Lodgepole Pine       0.89      0.88      0.88     84991
#   Ponderosa Pine       0.77      0.89      0.83     10726
#Cottonwood/Willow       0.84      0.61      0.70       824
#            Aspen       0.74      0.54      0.62      2848
#      Douglas-fir       0.71      0.60      0.65      5210
#        Krummholz       0.87      0.85      0.86      6153

#         accuracy                           0.86    174304
#        macro avg       0.81      0.75      0.77    174304
#     weighted avg       0.86      0.86      0.86    174304 
# 
# This by far our best model yet, could do some more tweaking to improve results in on our less-supported classes, but other than that just need some data viz tools and we can assemble our final model solution#


## Improved our performance even further by increasing trainable params
# Trainable Params: 16,472
# Epoch 11: early stopping
# Test loss: 0.29644957184791565
# Test accuracy: 0.8789758086204529

#                   precision    recall  f1-score   support

#       Spruce/Fir       0.89      0.87      0.88     63552
#   Lodgepole Pine       0.89      0.91      0.90     84991
#   Ponderosa Pine       0.83      0.89      0.86     10726
#Cottonwood/Willow       0.64      0.80      0.71       824
#            Aspen       0.75      0.56      0.64      2848
#      Douglas-fir       0.75      0.71      0.73      5210
#        Krummholz       0.93      0.85      0.89      6153

#         accuracy                           0.88    174304
#        macro avg       0.81      0.80      0.80    174304
#     weighted avg       0.88      0.88      0.88    174304