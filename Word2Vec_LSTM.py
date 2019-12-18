

import logging

import gensim
import nltk
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def createWord2VecModel(texts, targets, epochs_model):
    # WORD2VEC
    W2V_SIZE = 300
    W2V_WINDOW = 4
    W2V_EPOCH = 40
    W2V_MIN_COUNT = 8

    # KERAS
    SEQUENCE_LENGTH = 32
    EPOCHS = epochs_model
    BATCH_SIZE = 1024

    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
                                                window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT,
                                                workers=8)
    w2v_model.build_vocab(texts)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("Vocab size", vocab_size)

    train_x, test_x, train_y, test_y = train_test_split(texts, targets, test_size=0.20,
                                                        random_state=63)
    # train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.20, random_state=63)

    w2v_model.train(texts, total_examples=len(texts), epochs=W2V_EPOCH)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_x)
    vocab_size = len(tokenizer.word_index) + 1

    train_x_seq = pad_sequences(tokenizer.texts_to_sequences(train_x), maxlen=SEQUENCE_LENGTH)
    test_x_seq = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=SEQUENCE_LENGTH)

    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    # %%
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH,
                                trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.25))
    model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(1, activation='sigmoid'))
    #ACCURACY: 0.8107593655586243
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
        EarlyStopping(monitor='val_accuracy', min_delta=1e-2, patience=5)]

    history = model.fit(train_x_seq, train_y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=callbacks
                        )

    score = model.evaluate(test_x_seq, test_y, batch_size=BATCH_SIZE)
    print()
    print("ACCURACY:", score[1])
    print("LOSS:", score[0])


    return model



#    model = Sequential()
#    model.add(embedding_layer)
#    model.add(Dropout(0.25))
#    model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
#    model.add(Dropout(0.2))
#    model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
#    model.add(Dropout(0.2))
#    model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
#    model.add(Dropout(0.2))
#    model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1))
#    model.add(Dense(1, activation='sigmoid'))
#    #ACCURACY: 0.8107593655586243
