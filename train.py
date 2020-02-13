import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras import optimizers
import numpy as np
import random
import pickle
import efficientnet.keras as efn
from keras import backend as K

K.set_floatx('float16')
K.set_epsilon(1e-4)

num_of_class=7

# loading pretrained conv base model
conv_base = efn.EfficientNetB0(weights=None, include_top=False,input_shape=(128,128,3),classes=7)


dropout_rate = 0.2
model = Sequential()
model.add(conv_base)
model.add(GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(Dense(num_of_class, activation="softmax", name="fc_out"))

conv_base.trainable = True

rms = optimizers.RMSprop(lr=2e-5)

model.compile(
	optimizer=rms,
	loss="categorical_crossentropy",
	metrics=['accuracy'])

data_filename="sample2.dat"
with open(data_filename, "rb") as f:
    D = pickle.load(f)


dataset = D
random.shuffle(dataset)

train = dataset[:210]
test = dataset[210:]

X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

# Reshape for CNN input
#X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
#X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])

X_train = np.stack((X_train,)*3, axis=-1)
X_test = np.stack((X_test,)*3, axis=-1)


# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, num_of_class))
y_test = np.array(keras.utils.to_categorical(y_test, num_of_class))


##model = Sequential()
##input_shape=(128, 128, 1)
##
##model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
##model.add(MaxPooling2D((4, 2), strides=(4, 2)))
##model.add(Activation('relu'))
##
##model.add(Conv2D(48, (5, 5), padding="valid"))
##model.add(MaxPooling2D((4, 2), strides=(4, 2)))
##model.add(Activation('relu'))
##
##model.add(Conv2D(48, (5, 5), padding="valid"))
##model.add(Activation('relu'))
##
##model.add(Flatten())
###model.add(BatchNormalization())
##model.add(Dropout(rate=0.8))
##
##model.add(Dense(64))
###model.add(BatchNormalization())
##model.add(Activation('relu'))
##model.add(Dropout(rate=0.8))
##
##model.add(Dense(num_of_class))
##model.add(Activation('softmax'))

##model.compile(
##	optimizer="Adam",
##	loss="categorical_crossentropy",
##	metrics=['accuracy'])

model.fit(
	x=X_train, 
	y=y_train,
    epochs=2400,
    batch_size=2,
    validation_data= (X_test, y_test))

score = model.evaluate(
	x=X_test,
	y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
    
model.save("model2.hdf5")
