from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.datasets import imdb
import imdb
import numpy as np
from keras.preprocessing import text

# set parameters:
vocab_size = 1000
maxlen = 1000
batch_size = 32
embedding_dims = 50
filters = 128 #250
kernel_size = 3
hidden_dims = 100
epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()
print("Review", X_train[0])
print("Label", y_train[0])

tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print(X_train.shape)
print("After pre-processing", X_train[0])

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
model.add(Dropout(0.5))
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
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test), callbacks=[])
