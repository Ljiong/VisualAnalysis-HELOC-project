#Based on tutorial: https://machinelearningmastery.com/random-forest-ensemble-in-python/
#Run this code before you can classify

# Use numpy to convert to arrays
import numpy as np
import os
import joblib
from numpy import mean, std

# Pandas is used for data manipulation
import pandas as pd

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

X_test = []
def buildModel(features, labelDimension) :
    # Labels are the values we want to predict
    labels = np.array(features[labelDimension])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features= features.drop(labelDimension, axis = 1)

    # Convert to numpy array
    features = np.array(features)

    # Split the data into training and testing sets (heavily overfit on provided dataset to get as close as possible to the original model)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.30)
    X_test = train_features
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 10)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    #evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
    n_scores = cross_val_score(rf, features, labels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    print("done!")
    print("evaluating:")

    # report performance
    print(n_scores)
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    return rf

def main():
#load in the dataset
    features = pd.read_csv('./data/heloc_dataset_v1.csv')

    #the columns that stores the labels
    labelDimension = "RiskPerformance"

    #build a random forest classifier
    model = buildModel(features, labelDimension)
    dirs = 'SAVE_MODEL'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # 保存训练好的模型
    joblib.dump(model, dirs + '/trained_model.pkl')

# y_importance = model.feature_importances_
# print(y_importance)

if __name__ == "__main__":
    main()