import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from view import view

# logging code
layers = 3
hidden_layer_1_size = 128

first_layer_convs = 32
first_layer_conv_width = 3
first_layer_conv_height = 3
dropout = 0.4
dense_layer_size = 128
img_width = 28
img_height = 28
epochs = 10

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.
X_test = X_test / 255.

# Reshape


img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
          "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_train = X_train.astype('float32').reshape((-1,img_width,img_height,1))
X_test = X_test.astype('float32').reshape((-1,img_width,img_height,1))

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# create model
model = Sequential()
model.add(Conv2D(64,
    (first_layer_conv_width, first_layer_conv_height),
                 padding='same',
    input_shape=(28, 28,1),
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))
model.add(Conv2D(64,
    (first_layer_conv_width, first_layer_conv_height),
                 padding='same',
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(dense_layer_size, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])
# Fit the model

earlystopping = EarlyStopping(monitor='val_loss', 
                              min_delta=0, 
                              patience=11, 
                              verbose=1, 
                              mode='auto', 
                              baseline=None, 
                              restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=3, 
                              verbose=1, 
                              mode='auto', 
                              min_delta=0.0001, 
                              cooldown=0, 
                              min_lr=0.00001)

#keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', 
#                                verbose=0, 
#                                save_best_only=False, 
#                                save_weights_only=False, 
#                                mode='auto', 
#                                period=1)


history = model.fit(X_train, y_train, 
          epochs=epochs, 
          validation_data=(X_test, y_test),
          callbacks=[earlystopping,reduce_lr])

model.save('weights.hdf5')

view(history)