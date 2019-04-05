import numpy as np
import pandas
from sklearn.model_selection import train_test_split, GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(22, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model(optimizer='rmsprop', init='uniform', layer2_size = 8):
    # create model
    model = Sequential()
    model.add(Dense(22, input_dim=22, kernel_initializer=init, activation='relu'))
    model.add(Dense(layer2_size, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [1,10,50,100]
batches = [5,10,20,50]