import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import pickle


path = "C:\ZZZZZ IDHAR\Character-Recognization-System\DataSets\\Test\\"
files = os.listdir(path)[:]
classes1=[0,1,2,3,4,5,6,7,8,9]
classes=dict(zip(files,classes1))
print("Files ==>",files,"\n")

X_test=[]
Y_test=[]
for cl in classes:
    temp_path=path+cl
    for img_name in os.listdir(temp_path):
        img=cv2.imread(temp_path+"/"+img_name,0)
        X_test.append(img)
        Y_test.append(classes[cl])

X_test=tf.keras.utils.normalize(X_test,axis=1)



X_test=np.array(X_test)
Y_test= np.array(Y_test)

pickle_in = open("neuralmodel.pickle" ,"rb")

model = pickle.load(pickle_in)



prdictions = model.predict(X_test)
for i in range(0,10):
    plt.grid(False)
    plt.imshow(X_test[i],cmap=plt.cm.binary)
    plt.xlabel("Actual: " + files[Y_test[i]])
    plt.title("Predicted: "+ files[np.argmax(prdictions[i])])
    plt.show()
