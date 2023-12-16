## Forest Cover Classification project

## Using deep learning to predict forest cover type (the most common kind of tree cover) 
## Based only on cartographic variables.

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

y = data['class']

x = data[['Elevation',
          'Aspect',
          'Slope',
          'Horizontal_Distance_To_Hydrology',
          'Vertical_Distance_To_Hydrology',
          'Horizontal_Distance_To_Roadways',
          'Hillshade_9am',
          'Hillshade_Noon',
          'Hillshade_3pm',
          'Horizontal_Distance_To_Fire_Points',
          'Wilderness_Area1',
          'Wilderness_Area2',
          'Wilderness_Area3',
          'Wilderness_Area4',
          'Soil_Type1',
          'Soil_Type2',
          'Soil_Type3',
          'Soil_Type4',
          'Soil_Type5',
          'Soil_Type6',
          'Soil_Type7',
          'Soil_Type8',
          'Soil_Type9',
          'Soil_Type10',
          'Soil_Type11',
          'Soil_Type12',
          'Soil_Type13',
          'Soil_Type14',
          'Soil_Type15',
          'Soil_Type16',
          'Soil_Type17',
          'Soil_Type18',
          'Soil_Type19',
          'Soil_Type20',
          'Soil_Type21',
          'Soil_Type22',
          'Soil_Type23',
          'Soil_Type24',
          'Soil_Type25',
          'Soil_Type26',
          'Soil_Type27',
          'Soil_Type28',
          'Soil_Type29',
          'Soil_Type30',
          'Soil_Type31',
          'Soil_Type32',
          'Soil_Type33',
          'Soil_Type34',
          'Soil_Type35',
          'Soil_Type36',
          'Soil_Type37',
          'Soil_Type38',
          'Soil_Type39',
          'Soil_Type40']]
## One-Hot Label Encoding
x = pd.get_dummies(x)

## Step 3 - Splitting and Balancing

## train_test_split()
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.20, random_state=8)

## Column Transformer w/ StandardScaler() to tranform and scale our numeric feature columns to be properly interpreted by our model due to one-hot label encoding of our categorical variables
sc = StandardScaler()

## Using fit_transform() on x_train to fit and tranform data for our model
x_train = sc.fit_transform(x_train)

## Using .transform() on x_test to transform data for our model
x_test = sc.transform(x_test)


## Removed LabelEncoder() was not neccesary for model
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
    model.add(tf.keras.layers.Dropout(0.4))
    #model.add(Dense(64, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    
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
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)
## Fit Model
model.fit(x_train, y_train, epochs=10, batch_size=1000, verbose=1, validation_split=0.15, callbacks=[es])

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

## initial classification report

#  _warn_prf(average, modifier, msg_start, len(result))
            #  precision    recall  f1-score   support

          # 0       0.00      0.00      0.00     53025
          # 1       0.49      1.00      0.66     70754
          # 2       0.00      0.00      0.00      8956
          # 3       0.00      0.00      0.00       689
          # 4       0.00      0.00      0.00      2365
          # 5       0.00      0.00      0.00      4357
          # 6       0.00      0.00      0.00      5107

   # accuracy                           0.49    145253
  # macro avg       0.07      0.14      0.09    145253
#weighted avg       0.24      0.49      0.32    145253

# Class 1 is the only one returning results. Need to fix data and layers to help model predict for all possible classes. Also, will check out provided example to see where I can fix some of my code. 

## Classification report after reducing layers and epochs, increasing test size, and stratifying our labels
#              precision    recall  f1-score   support

         #  0       0.71      0.69      0.70     74144
         #  1       0.73      0.81      0.77     99156
         #  2       0.55      0.86      0.67     12514
         #  3       0.00      0.00      0.00       961
         #  4       0.00      0.00      0.00      3323
         #  5       0.00      0.00      0.00      6078
         #  6       0.85      0.26      0.39      7179

   # accuracy                           0.71    203355
  # macro avg       0.41      0.37      0.36    203355
#weighted avg       0.68      0.71      0.68    203355   
 
# We see improvement in our reports available data however there is still an imbalance in our data and we cannout properly test all possible results
# Will need to further balance our training and test data so that all possible results are at least relatively even
# Our overall accuracy has improved greatly as well as our macro and weighted averages 
 
 
# Last Classification Report for this starter model after viewing the solution and seeing some fixes for my code and issues

#Test loss: 0.4763410687446594
#Test accuracy: 0.8029913306236267 
# 
# 
#                   precision    recall  f1-score   support

      # Spruce/Fir       0.83      0.77      0.80     42368
  # Lodgepole Pine       0.80      0.89      0.84     56661
  # Ponderosa Pine       0.67      0.89      0.76      7151
#Cottonwood/Willow       0.00      0.00      0.00       549
  #          Aspen       0.84      0.19      0.31      1899
  #    Douglas-fir       0.63      0.13      0.21      3473
  #      Krummholz       0.87      0.74      0.80      4102

  #       accuracy                           0.80    116203
  #      macro avg       0.66      0.52      0.53    116203
  #   weighted avg       0.80      0.80      0.79    116203

## Overall we see a lot of improvement here in gather a report for all of our possibles classes, as well as improved accuracy and test loss. 
## However, we need to do further hyperparameter and data tuning in order to get a more complete model
## Then we shall put together some data vizualization for our model to better understand our model's performance and where it may be struggling/succeeding