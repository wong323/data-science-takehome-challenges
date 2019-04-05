
# coding: utf-8

# # Model Execution

# ## The Solution
# 
# Overall, this solution is designed to classify cases of fraudulent and non-fraudulent behaviour.
# 
# There are three models within this solution:
# 
# - One Extra Trees model, with cross-validation and RFE, in the sklearn_modelworkflow notebook.
# - One Baseline Sequential model written in Keras, in this notebook
# - One Sequential model with hyperparameter optimization, in this notebook. Note that this model takes an extremely long time to run depending on how you set the hyperparams!
# 
# todo: parallelize
# 
# 
# The structure of this code solution is as follows:
# 
/fraud_detector_v2a_1
├── keras_modelworkflow.ipynb
├── sklearn_modelworkflow.ipynb
├── modelconfig
│   ├── modelconfig.py
│   │   └── <model configuration code>
│   └── hyperparameters.json
└── inputparser.py
    ├── <data io functions>
    ├── <data cleaning functions>
    └── <data type conversion functions>
# ## The Script
# 
# 
# This script runs the Keras models by doing the following:
# 
# 1. Read in model and feature prep code
# 2. Initialize a model
# 3. Run that model to generate results.
# 
# 
# I implemented a baseline model, which runs with an input layer, a dropout later, then two dense layers to reduce the dimensionality of the representation to 1 output value. 
# 
# The baseline uses binary_crossentropy as the loss function, accuracy as the metric and adam as the performance score.
# 
# Accuracy makes sense in this context given that we wish to capture as much fraud activity, as accurately as possible. Binary cross entropy is a good performance measure in regression problems, so it makes sense here. Finally, Adam is an effective gradient descent optimizer, for more information see https://arxiv.org/abs/1412.6980.
# 

# In[23]:


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
from sklearn import preprocessing

import inputparser
from modelconfig.modelconfig import create_baseline, create_model 


# In[24]:


np.random.seed(7)


# Here we take advantage of our helper functionality to load in our dataset.

# In[26]:


traffic = inputparser.csv_reader('traffic.csv')
traffic = traffic.dropna()
y = traffic[['is_fraud']]
X = inputparser.x_maker(traffic)
X


# Now we're ready to try out a baseline classifier:

# In[27]:


# create an estimator
estimator = KerasClassifier(build_fn=create_baseline, batch_size=200, epochs=100, verbose=1)

#cross-validation with 10 folds
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# So far, so good! Next, let's adjust our model to perform HPO over the hyperparameter space using grid search.
# 
# Note that the operations being performed here can take an extended time.

# In[28]:


model = KerasClassifier(build_fn=create_model, verbose=2)

#this is an extensible set of hyperparams
optimizers = ['adam'] #https://keras.io/optimizers/
init = ['normal'] #https://keras.io/initializers/
epochs = [100] 
batches = [200]
layer2_size = [1, 2, 3, 4] 
#,8, 12
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init = init, layer2_size = layer2_size)

grid = GridSearchCV(estimator=model, cv = 10, param_grid=param_grid)

grid_result = grid.fit(X, y)


# In[29]:


#results on training data

means = grid_result.cv_results_['mean_train_score']
stds = grid_result.cv_results_['std_train_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("train: %f (%f) with: %r" % (mean, stdev, param))


# Ran the preexisting model over held out data, achieving passable results; 0.635 (0.045)

# In[30]:


#same results on held-out data
#one alternative might be to train on all the data in one go?

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("test: ""%f (%f) with: %r" % (mean, stdev, param))
    
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

