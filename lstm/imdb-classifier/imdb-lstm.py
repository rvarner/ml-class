from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, GRU
from keras.layers import Conv1D, Flatten, Bidirectional, MaxPooling1D
from keras.datasets import imdb
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import imdb
import numpy as np
from keras.preprocessing import text


wandb.init()
config = wandb.config

# set parameters:
vocab_size = 4000
maxlen = 300
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 128
epochs = 50

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(vocab_size,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2))
#model.add(LSTM(5,activation="sigmoid"))
model.add(Flatten())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("Model Summary:")
print(model.summary())

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=11, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback(),earlystopping,reduce_lr])
