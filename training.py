import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy
import pandas

model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(None, 28)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10))
print(model.summary())

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_validate, y_validate = x_test[:-10], y_test[:-10]
x_test, y_test = x_test[-10:], y_test[-10:]

CONST_TRAINING_SEQUENCE_LENGTH = 12
CONST_TESTING_CASES = 5


def dataNormalization(data):
    return [(datum - data[0]) / data[0] for datum in data]


def dataDeNormalization(data, base):
    return [(datum + 1) * base for datum in data]
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)	

model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1
)

for i in range(10):
    result = tf.argmax(model.predict(tf.expand_dims(x_test[i], 0)), axis=1)
    print(result.numpy(), y_test[i])

def getDeepLearningData(ticker):
    # Step 1. Load data
    data = pandas.read_csv('./data/Stock/' + ticker + '.csv')[
        'close'].tolist()
    # Step 2. Building Training data
    dataTraining = []
    for i in range(len(data) - CONST_TESTING_CASES * CONST_TRAINING_SEQUENCE_LENGTH):
        dataSegment = data[i:i + CONST_TRAINING_SEQUENCE_LENGTH + 1]
        dataTraining.append(dataNormalization(dataSegment))

    dataTraining = numpy.array(dataTraining)
    numpy.random.shuffle(dataTraining)
    X_Training = dataTraining[:, :-1]
    Y_Training = dataTraining[:, -1]

# Step 3. Building Testing data
    X_Testing = []
    Y_Testing_Base = []
    for i in range(CONST_TESTING_CASES, 0, -1):
        dataSegment = data[-(i + 1) * CONST_TRAINING_SEQUENCE_LENGTH:-i * CONST_TRAINING_SEQUENCE_LENGTH]
        Y_Testing_Base.append(dataSegment[0])
        X_Testing.append(dataNormalization(dataSegment))

    Y_Testing = data[-CONST_TESTING_CASES * CONST_TRAINING_SEQUENCE_LENGTH:]

    X_Testing = numpy.array(X_Testing)
    Y_Testing = numpy.array(Y_Testing)

    # Step 4. Reshape for deep learning
    X_Training = numpy.reshape(X_Training, (X_Training.shape[0], X_Training.shape[1], 1))
    X_Testing = numpy.reshape(X_Testing, (X_Testing.shape[0], X_Testing.shape[1], 1))

    return X_Training, Y_Training, X_Testing, Y_Testing, Y_Testing_Base


def predict(model, X):
    predictionsNormalized = []

    for i in range(len(X)):
        data = X[i]
        result = []

        for j in range(CONST_TRAINING_SEQUENCE_LENGTH):
            predicted = model.predict(data[numpy.newaxis, :, :])[0, 0]
            result.append(predicted)
            data = data[1:]
            data = numpy.insert(data, [CONST_TRAINING_SEQUENCE_LENGTH - 1], predicted, axis=0)

        predictionsNormalized.append(result)

    return predictionsNormalized
def predictLSTM(ticker):
    # Step 1. Load data
    X_Training, Y_Training, X_Testing, Y_Testing, Y_Testing_Base = getDeepLearningData(ticker)

    # Step 2. Build model
    #model = Sequential()

    model.add(LSTM(
        input_shape=(None, 1),
        units=50,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        200,
        return_sequences=False))
    model.add(Dropout(0.2))

    #model.add(Dense(units=1))
    #model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')
    # Step 3. Train model
    model.fit(X_Training, Y_Training,
              batch_size=512,
              epochs=27,
              validation_split=0.05)

    # Step 4. Predict
    predictionsNormalized = predict(model, X_Testing)

    # Step 5. De-nomalize
    predictions = []
    for i, row in enumerate(predictionsNormalized):
        predictions.append(dataDeNormalization(row, Y_Testing_Base[i]))



model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
    
    # returns a compiled model
    # identical to the previous one
model = load_model('my_model.h5')
predictLSTM(ticker='IBM')