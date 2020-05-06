import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

INPUT_DATA_SIZE = 3  # we use 3 columns from CSV as input data
OUTPUT_DATA_SIZE = 2  # one number as an output data
ACTIONS = ["run", "attack"] # type of actions


def create_dataset(csv_file):
    # 1. create train data

    # columns for train data
    train_data = pd.read_csv(csv_file, usecols=["enemyNearby", "hasAmmo", "health"]).values

    # target data with compared to train data
    target_data = pd.read_csv(csv_file, usecols=["run", "attack"]).values

    # Reshape and normalize training data
    train_x = train_data[:].reshape(train_data.shape[0], INPUT_DATA_SIZE).astype('int32')
    # you can chose the type of data: int32, float32, etc
    targets_x = target_data[:].reshape(target_data.shape[0], OUTPUT_DATA_SIZE).astype('int32')

    # 2. train model

    model = create_model()
    # train your model
    model.fit(train_x, targets_x, epochs=10, batch_size=5)

    # 3. use trained model with real input
    input_test_data_to_model(model, 1, 1, 90)


def input_test_data_to_model(model, enemyNearby, hasAmmo, health):
    test_input_data = np.array([[enemyNearby, hasAmmo, health]])

    # (optional) prepare input data for trained module. In our case we can use `test_input_data` in model.predict without reshape
    preapared_test_data = test_input_data[:].reshape(test_input_data.shape[0], INPUT_DATA_SIZE).astype('int32')

    # result actions for this test data
    predictions = model.predict(preapared_test_data)

    # get exactly one action that more suit for prediction
    win_prediction = np.argmax(predictions)
    print(win_prediction)
    print("-> %s (%s)" % (ACTIONS[win_prediction], predictions.tolist()))


def create_model():
    model = Sequential()

    # input layer
    model.add(Dense(INPUT_DATA_SIZE, input_dim=INPUT_DATA_SIZE, activation='relu'))
    # hidden layer
    model.add(Dense(16, activation='relu'))
    # output layer
    model.add(Dense(OUTPUT_DATA_SIZE, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #show strucutre
    print(model.summary())

    return model


create_dataset("../simple-character-ai/dataset.csv")
