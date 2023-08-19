"""
Machine Learning Modeling for Starbucks Capstone Project

This script contains functions and procedures to train a machine learning model
that predicts whether a customer will complete an offer based on their demographics
and offer details.

Functions:
    prepare_data_for_modeling: Prepares the dataset for modeling.
    train_model: Trains a Logistic Regression model.
    evaluate_model: Evaluates the trained Logistic Regression model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_preparation import preprocess_data


def prepare_data_for_modeling(merged_data):
    """
    Prepare the dataset for modeling.
    
    Args:
        merged_data (DataFrame): The merged dataset.
    
    Returns:
        tuple: Features and labels for training and testing datasets.
    """
    merged_data['offer_completed'] = merged_data['event'].apply(lambda x: 1 if x == 'offer completed' else 0)
    dummies_offer_type = pd.get_dummies(merged_data['offer_type'], prefix='offer')
    dummies_gender = pd.get_dummies(merged_data['gender'], prefix='gender')
    merged_data = pd.concat([merged_data, dummies_offer_type, dummies_gender], axis=1)

    features = merged_data[['age', 'income', 'offer_bogo', 'offer_discount', 'offer_informational', 'gender_F', 'gender_M', 'gender_O', 'difficulty', 'reward', 'duration']]
    labels = merged_data['offer_completed']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(merged_data):
    """
    Train a Logistic Regression model.
    
    Args:
        merged_data (DataFrame): The merged dataset.
    
    Returns:
        tuple: Trained model, X_test, y_test, and X_train datasets.
    """
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(merged_data)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    
    return lr_model, X_test, y_test, X_train


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained Logistic Regression model.
    
    Args:
        model (LogisticRegression): The trained model.
        X_test (DataFrame): The test features.
        y_test (Series): The test labels.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, predictions))


if __name__ == '__main__':
    portfolio, profile, merged_data = preprocess_data()
    lr_model, X_test, y_test, X_train = train_model(merged_data)
    evaluate_model(lr_model, X_test, y_test)
