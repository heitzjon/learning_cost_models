import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense, Embedding, Flatten, Conv1D, Input, Lambda
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (7.5,5.5)


DATA_PATH = "data_queries_postgres_stat_synt.txt"
MODEL_PATH = "models/costEstimation_onehot_190117.h5"

# The available models are:
#       'onehot': query is transformed to a one hot encoded vector for the input of a dense layer
#       'embedding': query gets fed into a embedding layer, then processed through a convolutional layer
MODEL = 'onehot'

# Target values are:
#       'exec_time': measured execution time of the queries in postgres
#       'act_row': the resulting row count of the queries
COLUMN = 'exec_time'


# ONLY FOR ONEHOT-MODEL: The modes available include:
#     ‘binary‘: Whether or not each word is present in the document. This is the default.
#     ‘count‘: The count of each word in the document.
#     ‘tfidf‘: The Text Frequency-Inverse DocumentFrequency (TF-IDF) scoring for each word in the document.
#     ‘freq‘: The frequency of each word as a ratio of words within each document.
TEXT_ENCODING_MODE = "binary"


def costestimator_model(allquery,queryset,costs,norm_factor):
    # fix random seed for reproducibility
    np.random.seed(7)

    #PREPROCESSING X
    t = Tokenizer()
    t.fit_on_texts(allquery)
    X = t.texts_to_matrix(queryset, mode=TEXT_ENCODING_MODE)

    #PREPROCESSING Y
    Y=costs/norm_factor

    # create model
    model = Sequential()
    model.add(Dense(450, input_dim=len(X[0]), activation='relu')) #shape is 49
    model.add(Dense(45, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1))

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'msle'])
    print(model.summary())

    # Fit the model
    hist = model.fit(X, Y, validation_split=0.33, epochs=250, batch_size=840)

    plt.plot(hist.history['mean_squared_error'])
    #plt.plot(hist.history['mean_absolute_error'])
    #plt.plot(hist.history['mean_absolute_percentage_error'])
    #plt.plot(hist.history['mean_squared_logarithmic_error'])
    plt.show()

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model

def costestimator_model_embedding(allquery,queryset,costs,norm_factor):
    # fix random seed for reproducibility
    np.random.seed(7)

    #PREPROCESSING X
    t = Tokenizer()
    t.fit_on_texts(allquery)
    encoded_docs = t.texts_to_sequences(queryset)
    max_length = allquery.map(lambda x: len(x)).max()
    X = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    #PREPROCESSING Y
    Y=costs/norm_factor

    # create model
    model = Sequential()
    model.add(Embedding(len(X[0]), 32, input_length=max_length))
    model.add(Conv1D(16, (8), activation='relu'))
    model.add(Flatten())
    model.add(Dense(45, activation='relu'))
    model.add(Dense(1))

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'msle'])

    print(model.summary())

    # Fit the model
    hist = model.fit(X, Y, validation_split=0.33, epochs=250, batch_size=840)

    plt.plot(hist.history['mean_squared_error'])
    #plt.plot(hist.history['mean_absolute_error'])
    #plt.plot(hist.history['mean_absolute_percentage_error'])
    #plt.plot(hist.history['mean_squared_logarithmic_error'])
    plt.show()

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model


def train():
    dataset = pd.read_csv(DATA_PATH,sep="|")

    print(list(dataset))
    
    train = dataset.sample(frac=0.75, random_state=99)
    # you can't simply split 0.75 and 0.25 without overlapping
    # this code tries to find that train = 75% and test = 25%
    test = dataset.loc[~dataset.index.isin(train.index), :]


    y_norm_factor=dataset[COLUMN].max()
    if MODEL is 'onehot':
        m = costestimator_model(dataset["query"],train["query"],train[COLUMN],y_norm_factor)
    else:
        m = costestimator_model_embedding(dataset["query"],train["query"],train[COLUMN],y_norm_factor)
    m.save(MODEL_PATH,overwrite=False)



    t = Tokenizer()
    t.fit_on_texts(dataset["query"])

    if MODEL is 'onehot':
        train_x = t.texts_to_matrix(train["query"], mode=TEXT_ENCODING_MODE)
        test_x = t.texts_to_matrix(test["query"], mode=TEXT_ENCODING_MODE)
    else:
        max_length = dataset["query"].map(lambda x: len(x)).max()
        encoded_docs = t.texts_to_sequences(train["query"])
        train_x = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        encoded_docs = t.texts_to_sequences(test["query"])
        test_x = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


    train_y=m.predict(train_x)*y_norm_factor
    test_y=m.predict(test_x)*y_norm_factor

    if COLUMN is 'exec_time':
        plt.xlabel('estimated execution time')
        plt.ylabel('execution time by postgres')
    else:
        plt.xlabel('predicted # of rows')
        plt.ylabel('actual # of rows')
        plt.scatter(dataset['est_row'], dataset[COLUMN], c='yellow')

    plt.scatter(train_y, train[COLUMN], c='blue')
    plt.scatter(test_y, test[COLUMN], c='red')
    plt.show()

def test():
    dataset = pd.read_csv(DATA_PATH, sep="|")
    m = load_model(MODEL_PATH)

    train = dataset.sample(frac=0.75, random_state=99)
    test = dataset.loc[~dataset.index.isin(train.index), :]

    y_norm_factor = dataset[COLUMN].max()

    t = Tokenizer()
    t.fit_on_texts(dataset["query"])

    if MODEL is 'onehot':
        train_x = t.texts_to_matrix(train["query"], mode=TEXT_ENCODING_MODE)
        test_x = t.texts_to_matrix(test["query"], mode=TEXT_ENCODING_MODE)
    else:
        max_length = dataset["query"].map(lambda x: len(x)).max()
        encoded_docs = t.texts_to_sequences(train["query"])
        train_x = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        encoded_docs = t.texts_to_sequences(test["query"])
        test_x = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    train_y = m.predict(train_x) * y_norm_factor
    test_y = m.predict(test_x) * y_norm_factor

    if COLUMN is 'exec_time':
        plt.xlabel('estimated execution time')
        plt.ylabel('execution time')
    else:
        plt.xlabel('predicted # of rows')
        plt.ylabel('actual # of rows')

    plt.scatter(train_y, train[COLUMN], c='blue', label='training data',s=10)
    plt.scatter(test_y, test[COLUMN], c='red', label='test data',s=10)
    plt.legend()
    plt.show()

    df_pq = pd.DataFrame()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    if COLUMN is "act_row": df_pq['diff'] = dataset["est_row"] - dataset[COLUMN]
    df_train['diff'] = train_y[:, 0] - train[COLUMN]
    df_test['diff'] = test_y[:, 0] - test[COLUMN]

    # plot the histogram
    if COLUMN is "act_row":
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].hist(df_pq['diff'], bins=100, color='yellow')
        axarr[0].set_title("difference between predicted and actual cardinality")
        axarr[1].hist(df_train['diff'], bins=100, color='blue')
        axarr[2].hist(df_test['diff'], bins=100, color='red')
        axarr[1].set_ylabel("# samples")
        axarr[2].set_xlabel("dff in # rows")

    else:
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].set_title("difference between predicted and actual cardinality")
        axarr[0].hist(df_train['diff'], bins=100, color='blue')
        axarr[1].hist(df_test['diff'], bins=100, color='red')
        axarr[0].set_ylabel("# samples")
        axarr[1].set_xlabel("dff in # rows")

    plt.show()


#train()
#test()
