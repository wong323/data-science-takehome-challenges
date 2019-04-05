
# coding: utf-8

# # Model Execution

# ## The Solution
# 
# Overall, this solution is designed to classify cases of fraudulent and non-fraudulent behaviour.
# 
# There are three models within this solution:
# 
# - One Extra Trees model, with cross-validation and RFE, in this notebook.
# - One Baseline Sequential model written in Keras, in the keras_modelworkflow notebook
# - One Sequential model with hyperparameter optimization, in the keras_modelworkflow notebook.
# 
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
# This script runs the sklearn model by doing the following:
# 
# 1. Read in model and feature prep code
# 2. Initialize a model
# 3. Run that model to generate results.
# 
# 
# I implemented an extra trees classifier, and employed cross-validation and recursive feature elimination approaches.
# 
# One objective of this work was to quickly identify feature effectiveness.

# In[83]:


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE, RFECV, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy
from numpy.random import standard_normal
import inputparser
# fix random seed for reproducibility
numpy.random.seed(7)


# Here we take advantage of our helper functionality to load in our dataset.

# In[85]:



traffic = inputparser.csv_reader('traffic.csv')
traffic = traffic.dropna()
#traffic = traffic[traffic['number_of_mouse_clicks']==True]
print(traffic.shape)
Y = traffic[['is_fraud']]
x = inputparser.x_maker(traffic)
x


# In[86]:


Y = Y.values.ravel()
Y


# In[90]:


get_ipython().run_cell_magic('time', '', "\nclf = etc(n_estimators = 240, random_state = 0)\n#rfe = RFE(estimator=clf, n_features_to_select=1, step=1)\nrfecv = RFECV(estimator=clf, step=0.1, cv=StratifiedKFold(2), n_jobs=4,\n              scoring='accuracy', verbose = 2)\n\nrfecv.fit(x, Y)")


# In[91]:


plt.figure()
plt.xlabel("# features selected")
plt.ylabel("Cross validation score (prop. of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[92]:


rfecv.ranking_


# In[87]:


get_ipython().run_cell_magic('time', '', 'clf.fit(x, Y)\nimportances = clf.feature_importances_\nstd = numpy.std([tree.feature_importances_ for tree in clf.estimators_],\n             axis=0)\nindices = numpy.argsort(importances)[::-1]')


# In[88]:


plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.show()


# In[89]:


print("Feature ranking:")
for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

