

# Importing packages
import sys
import json
import math
import numpy as np
import pandas as pd
import datetime,time
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#model = RandomForestRegressor(max_depth=30, random_state=2)
#model.fit(X_train, y_train)
#joblib.dump(model, 'RFR+model_for_amin.pkl')

print("It is working")
