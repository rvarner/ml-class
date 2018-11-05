from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, ZeroPadding2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras.optimizers import Adam


def create_run_log_dir(run_name=None, log_dir='/notebooks/tf_logs'):
    import os
    import os.path
    name = os.getcwd().split('/')[-1]
    if run_name != None:
        name = '%s-%s' % (name, run_name)
    i = 1
    while os.path.isdir(os.path.join(log_dir,'%s-%d' % (name,i))):
        i += 1
    
    name = '%s-%d' % (name, i)                    
    print('Using run_name = %s' % name)            
    path = os.path.join(log_dir,name)
    os.makedirs(path, 755)
    return path
        

first_layer_convs = 32
first_layer_conv_width = 3
first_layer_conv_height = 3
dropout = 0.5
dense_layer_size = 128
filter_count = 128
img_width = 28
img_height = 28
epochs = 50


#(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
labels=range(10)

# build model

model = Sequential()
model.add(Conv2D(filter_count,
                 (first_layer_conv_width, first_layer_conv_height),
                 padding='same',
                 input_shape=(28, 28,1),
    activation='relu'))
model.add(Conv2D(filter_count,
                 (first_layer_conv_width, first_layer_conv_height),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(dropout))
model.add(Conv2D(filter_count,
                 (first_layer_conv_width, first_layer_conv_height),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filter_count,
                 (first_layer_conv_width, first_layer_conv_height),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(dropout))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=3, 
                              verbose=1, 
                              mode='auto', 
                              min_delta=0.0001, 
                              cooldown=0, 
                              min_lr=0.00001)

earlystopping = EarlyStopping(monitor='val_loss', 
                              min_delta=0, 
                              patience=10, 
                              verbose=1, 
                              mode='auto', 
                              baseline=None, 
                              restore_best_weights=True)

model.fit(X_train,
          y_train, 
          validation_data=(X_test, y_test),
          epochs=epochs,
          callbacks=[
              reduce_lr,
              earlystopping,
              TensorBoard(log_dir=create_run_log_dir('2xccmp%d-256-64-fashion' % (filter_count)))])

