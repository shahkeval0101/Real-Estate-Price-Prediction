# CodeWithHarry Project on real estate price prediction
# Supervised Learning
# Regression model
# Batch learning
# Accuracy RMSE, (Mean Absolute Error, Manhattan Norm)
# See notebook from Harry if doubt



# import lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# dataset taken from uci repository
# import dataset
housing = pd.read_csv(r"C:\Users\keval\OneDrive\Desktop\machine learning course learning and important material\Machine Learning Project\Real_price_prediction\Data.csv")
housing.shape
housing.head()
housing.tail()
housing.info()
housing.describe()

housing.hist(bins = 50, figsize = (20, 15))#value vs freq 
plt.show()


"""
for learning purpose only
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
"""

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size =  0.2, random_state = 42)

# using stratified sampling will help in taking all variety of values in our training set in a way it should represent our entire population
# Stratified sampling should be done on most important feature
# This should be done on one column here we have choosen CHAS attribute


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)


for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


#is Charles River dummy variable 

strat_train_set["CHAS"].value_counts()
strat_test_set["CHAS"].value_counts()


housing = strat_train_set.copy() # so this step has copied my train set to housing so I can use housing everywhere which plot on my training data only and not touch test set


# Looking for correlation
# so this is a pearson coefficient, we remove it wrt to other columns to know that if we increase other feature value what will happen to this value
# for eg if we increase RM value MEDV will increase as it is strongly positive related and if we increase LSTAT MEDV will decrease
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# This plot helps to visualize the correlation if slope is positive between attributes so they are strongly related to each other
from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))

housing.plot(kind = "scatter", x = "RM", y = "MEDV", alpha = 0.8)


# Try out some combination with dataset so that it improves the ML model
housing["TAXRM"] = housing["TAX"]/housing["RM"]
housing.head()

# see the correlation with how this featuere affects it
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)
# so on adding this TAXRM feature i.e if TAXRM increases than MEDV decreses

housing.plot(kind = "scatter", x= "TAXRM", y= "MEDV", alpha = 0.8)



# Making X_train and y_train
housing = strat_train_set.drop("MEDV", axis=1) # this train_set doesnot include in our new TAXRM column and also we have removed last column out of it
housing_labels = strat_train_set["MEDV"].copy() # here we have extracted last column as label
housing.shape # this is our X_train matrix
housing_labels.shape # this is our Y_train vector 
# We will put this in our model fit method


"""
# on having missing attributes
# 1.remove row
# 2.Take mean,mean,zero
# get rid of column

# original value wont be changed
a = housing.dropna(subset  = ["RM"])
a.shape()

housing.drop("RM",axis = 1)
# RM column is dropped

# to fill with mean
housing.fillna(housing["RM"].mean())
# to fill with median
housing.fillna(housing["RM"].median())
"""

# We need to fit this mean or median value to train set, test set, or any new data that is coming so we use imputer class
# imputer helps in filling missing value

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)
imputer.statistics_

X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns = housing.columns)
housing_tr.describe


"""
Sckit learn design
Primarily there are three type of object i
Estimators - It estimates some paramerter based on the dataset eg=Imputer
it has fit and transform method
fit method - fits dataset and calculates the parameter
strategy is hyper parameter

Transformer - takes input and returns o/p based on the learning from fit().
IT also convenience function called fit_transform

predictors - Linear regressor is example of predictors
fit method
predict method
score function which evaluates the predict
"""


"""
Feature scaling
Primarily two types
Min-Max scaling (Normalization)
(value - min)/(max - min) #0-1 values
sklearn provides MinMaxScaler for this



Standardization(better one)
(value - mean)/std
StandardScalar class for this
"""
 
# Creating a pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = "median")),
        ('standard_Scalar', StandardScaler())
    ])
housing_num_tr = my_pipeline.fit_transform(housing_tr)
# housing_num_tr contains imputation for missing values and feature scaling so it will be used as X_train

"""
simpleImputer = calculates missing value with strategy and then puts in that column
it has two methods to do it:
fit method calculates mean/median based on the strategy
transform method replaces this missing value
This is only in case of SimpleImputer


For standardScalar
Feature scaling is a technique to standardize the independent features present in the data to fix range. It is performed during data preprocessing step to handle varying magnitudes, units or values. if feature scaling is not done then machine learning algo tends to weigh greater values higher and smaller values lower regardless of units and values
For eg:it will treat 3000m higher than 5km that's actually not true so it can give wrong predictions
so feature scaling needs to be done

fit method calculates : mean and sigma 
This is will help to calculate for train_Set, test_Set, and any real time coming values
transform  applies transforamtion to any dataset


"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)





# Evaluation the matrix
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
rmse


# using better evaluation technique
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores




def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

print_scores(rmse_scores)


# Saving the model
from joblib import dump, load
dump(model, 'Dragon.joblib')


# Testing the model 
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

prepared_data[0]


# Using the model
from joblib import dump, load
import numpy as np
model = load('Dragon.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)






