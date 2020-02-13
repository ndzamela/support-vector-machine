# --------------
# Loading the Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset


# Check the correlation between each feature and check for null values


# Print total no of labels also print number of Male and Female labels


# Label Encode target variable


# Scale all the independent features and split the dataset into training and testing set.


# Build model with SVC classifier keeping default Linear kernel and calculate accuracy score.


# Build SVC classifier model with polynomial kernel and calculate accuracy score


# Build SVM model with rbf kernel.


#  Remove Correlated Features.


# Split the newly created data frame into train and test set, scale the features and apply SVM model with rbf kernel to newly created dataframe


# Do Hyperparameter Tuning using GridSearchCV and evaluate the model on test data.





