from keras.preprocessing import image
import matplotlib.pyplot as plt
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
import os
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import load_model




def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(24,24)
train_batch = generator("data/train_set",shuffle=True,batch_size=BS,target_size=TS)
valid_batch = generator("data/test_set",shuffle=True,batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS



model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),



    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

H = model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)

model.save('drowsinessDetection.h5', overwrite=True)


plt.figure(figsize=(14,9))

epochs = H.epoch

val = H.history["val_accuracy"]
train = H.history["accuracy"]

plt.plot(epochs,val,color="blue",linestyle = "solid",label="validation accuracy")
plt.plot(epochs,train,color="blue",linestyle = "dashed",label="train accuracy")

plt.title("accuracy model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(np.arange(20))
plt.legend()
plt.savefig('custom_model.svg')
plt.show()