import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2
import pickle

# Importing Datasets 
path = "C:\\ZZZZZ IDHAR\\Character-Recognization-System\\DataSets"
files = os.listdir(path)[:]
classes1=[0,1,2,3,4,5,6,7,8,9]
classes=dict(zip(files,classes1))
print("Files ==>",files,"\n")
print("Classes ==>",classes)

# Preparing the dataset
X_train=[]
Y_train=[]
for cl in classes:
    temp_path=path+cl
    for img_name in os.listdir(temp_path):
        img=cv2.imread(temp_path+"/"+img_name,0)
        X_train.append(img)
        Y_train.append(classes[cl])
pd.Series(Y_train).value_counts()

#Splitting the dataset into Training set And Testing set  Using Train and test split function 
X_train, X_test, Y_train, Y_test= train_test_split(X_train,Y_train,test_size=.20,random_state=42)

# Converting the  lists into Array using Numpy 
X_train= np.array(X_train)
Y_train= np.array(Y_train)
Y_test=np.array(Y_test)
Y_train= np.array(Y_train)

# Normalizaing the Datasets  For better efficiency
X_train=tf.keras.utils.normalize(X_train,axis=1)
X_test=tf.keras.utils.normalize(X_test,axis=1)

# Creating the Model 
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(32,32)))
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# Training the Model
model.fit(X_train,Y_train,epochs=25)


# Saving the model (for Future use ) so that we dont have to train the model again and again Uing pickle

with open("neuralmodel.pickle","wb") as f:
    pickle.dump(model,f)  


# Predicting the data from  the testing Dataset 
prdictions = model.predict(X_test)
for i in range(0,10):
    plt.grid(False)
    plt.imshow(X_test[i],cmap=plt.cm.binary)
    plt.xlabel("Actual: " + files[Y_test[i]])
    plt.title("Predicted: "+ files[np.argmax(prdictions[i])])
    plt.show()
