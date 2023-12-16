


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
## class is our tree cover-type classification ranging from 1-7 (our label)
## all other variables will be our features and all can be included in model

## Step Two - Preprocessing

## Separate into labels(y) and features (x)

y = data.iloc[:, -1]

x = data.iloc[:, 0:-1]

## One-Hot Label Encoding
x = pd.get_dummies(x)

## Step 3 - Splitting and Balancing

## train_test_split()
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.33, random_state=60)

## StandardScaler() scale our numeric feature columns to be properly interpreted by our model
sc = StandardScaler()

## Using fit_transform() on x_train to fit and tranform data for our model
x_train = sc.fit_transform(x_train)

## Using .transform() on x_test to transform data for our model
x_test = sc.transform(x_test)

## Step 4 - designing and building our model

def design_model(num_features):
    # Sequential()
    model = Sequential()
    # Input Layer
    model.add(InputLayer(input_shape=(x_train.shape[1], )))
    # Hidden Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    # Output Layer
    model.add(Dense(8, activation='softmax'))
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    # Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

## 16,472 trainable parameters

## Call model on training data
columns = data.columns.tolist()
features, label = columns[:-1], columns[-1]

num_features = len(features)
model = design_model(num_features)

## Model Summary
print('Summary of Model:')
model.summary()
## Early Stopping here
es = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
## Fit Model
history = model.fit(x_train, y_train, epochs=12, batch_size=1000, verbose=1, validation_split=0.15, callbacks=[es])

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

# Report:
# Trainable Params: 16,472
# Epoch 11: early stopping
# Test loss: 0.31046950817108154
# Test accuracy: 0.8715876936912537

#                   precision    recall  f1-score   support
#
#       Spruce/Fir       0.89      0.84      0.87     69907
#   Lodgepole Pine       0.87      0.92      0.90     93489
#   Ponderosa Pine       0.86      0.81      0.84     11799
#Cottonwood/Willow       0.78      0.63      0.70       907
#            Aspen       0.80      0.52      0.63      3133
#      Douglas-fir       0.66      0.75      0.70      5731
#        Krummholz       0.90      0.86      0.88      6768
#
#         accuracy                           0.87    191734
#        macro avg       0.82      0.76      0.79    191734
#     weighted avg       0.87      0.87      0.87    191734


## Data Vizualition
def plot_history(history, param):
    if param == 'acc':
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    elif param == 'loss':
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()

plot_history(history, 'acc')
plot_history(history, 'loss')

## Confusion Matrix with Heatmap
def plot_heatmap(class_names, y_estimate, y_test):
    cm = confusion_matrix(y_test, y_estimate)
    fig, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(cm, fmt='g', cmap='Blues', annot=True, ax=ax)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    # Save the heatmap to file
    heatmapfig = heatmap.get_figure()
    
    plt.show()

plot_heatmap(class_names, y_estimate, y_test)
