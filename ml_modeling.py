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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from data_preparation import preprocess_data
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def prepare_data_for_modeling(merged_data):
    """
    Prepare the dataset for modeling.

    Args:
        merged_data (DataFrame): The merged dataset.

    Returns:
        tuple: Features and labels for training and testing datasets.
    """
    print(merged_data.columns)  # Debugging line to print available columns

    # Create dummy variables
    merged_data['offer_completed'] = merged_data['event'].apply(lambda x: 1 if x == 'offer completed' else 0)
    dummies_offer_type = pd.get_dummies(merged_data['offer_type'], prefix='offer')
    dummies_gender = pd.get_dummies(merged_data['gender'], prefix='gender')
    merged_data = pd.concat([merged_data, dummies_offer_type, dummies_gender], axis=1)

    # Prepare features and labels
    features = merged_data[
        ['age', 'income', 'offer_bogo', 'offer_discount', 'offer_informational', 'gender_F', 'gender_M', 'gender_O',
         'difficulty', 'reward', 'duration']]
    labels = merged_data['offer_completed']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Handle NaN values by replacing them with the median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    return X_train_imputed, X_test_imputed, y_train, y_test

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
    from data_preparation import load_data  # Add this import at the top

    portfolio, profile, transcript = load_data()
    merged_data = preprocess_data(portfolio, profile, transcript)
    lr_model, X_test, y_test, X_train = train_model(merged_data)
    evaluate_model(lr_model, X_test, y_test)

    # Plotting confusion matrix manually
    cm = confusion_matrix(y_test, lr_model.predict(X_test))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()