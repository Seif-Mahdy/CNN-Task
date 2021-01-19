from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.optimizers import Adam


# load dataset
def load_dataset():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # reshape dataset to have a single channel
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    # one hot encode target values
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_x, train_y, test_x, test_y


def prepare_pixels(train_dataset, test_dataset):
    train_dataset = train_dataset.astype('float32')
    test_dataset = test_dataset.astype('float32')
    train_dataset /= 255.0
    test_dataset /= 255.0
    return train_dataset, test_dataset


# shuffle training data
def shuffle_training_data(train_x, train_y):
    combined = list(zip(train_x, train_y))
    np.random.shuffle(combined)
    train_x[:], train_y[:] = zip(*combined)
    return train_x, train_y


# define cnn model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# define cnn model
def create_model_with_adam_optimizer():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# define cnn model with batch normalization
def create_model_batch_normalization():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# define cnn model with changing the capacity of the feature extraction
def create_model_increase_capacity():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(data_x, data_y, model, n_folds=5):
    accuracies = list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(data_x):
        # select rows for train and test
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]
        # fit model
        history = model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y), verbose=0)
        # evaluate model
        _, acc = model.evaluate(test_x, test_y, verbose=0)
        accuracies.append(acc)
    return accuracies


# run the test harness for evaluating a model
def run_test():
    accuracies = list()

    # load dataset
    train_x, train_y, test_x, test_y = load_dataset()

    # shuffle training set
    train_x, train_y = shuffle_training_data(train_x, train_y)

    # prepare pixel data
    train_x, test_x = prepare_pixels(train_x, test_x)

    # evaluate model
    model1 = create_model()
    model2 = create_model_batch_normalization()
    model3 = create_model_increase_capacity()
    model4 = create_model_with_adam_optimizer()

    accuracies.append(evaluate_model(train_x, train_y, model1))
    accuracies.append(evaluate_model(train_x, train_y, model2))
    accuracies.append(evaluate_model(train_x, train_y, model3))
    accuracies.append(evaluate_model(train_x, train_y, model4))

    for i in range(len(accuracies)):
        for j in range(len(accuracies[i])):
            if j == 0:
                print('Model: ', i)
            print(accuracies[i][j] * 100)


# entry point, run the test harness
run_test()
