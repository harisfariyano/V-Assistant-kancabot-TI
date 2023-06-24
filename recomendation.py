from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Flatten, MaxPooling2D , Conv2D, Dropout, GlobalAveragePooling2D
import tensorflow as tf

def make_model():
    mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=(224 , 224, 3),include_top=False,weights='imagenet')
    model_gender = Sequential()
    model_gender.add(mobilenet)
    model_gender.add(GlobalAveragePooling2D())
    model_gender.add(Flatten())
    model_gender.add(Dense(1024, activation="relu"))
    model_gender.add(Dense(512, activation="relu"))
    model_gender.add(Dense(2, activation="softmax" , name="classification"))
   
    model_gender.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005,momentum=0.9), 
            loss='categorical_crossentropy', 
            metrics = ['accuracy'])
    return model_gender