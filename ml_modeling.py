import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_preparation import load_data, preprocess_data

# Load data
portfolio, profile, transcript = load_data()

# Preprocess data
merged_data = preprocess_data(portfolio, profile, transcript)


# Assuming merged_data is already loaded and prepared as done previously
def prepare_data_for_modeling(merged_data):
    # Create a binary label for offer completion
    merged_data['offer_completed'] = merged_data['event'].apply(lambda x: 1 if x == 'offer completed' else 0)

    # Convert offer type and gender to dummy variables
    dummies_offer_type = pd.get_dummies(merged_data['offer_type'], prefix='offer')
    dummies_gender = pd.get_dummies(merged_data['gender'], prefix='gender')
    merged_data = pd.concat([merged_data, dummies_offer_type, dummies_gender], axis=1)

    # Extract features and labels
    features = merged_data[['age', 'income', 'offer_bogo', 'offer_discount', 'offer_informational', 'gender_F', 'gender_M', 'gender_O', 'difficulty', 'reward', 'duration']]
    labels = merged_data['offer_completed']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Handling NaN values
    X_train['difficulty'].fillna(0, inplace=True)
    X_train['reward'].fillna(0, inplace=True)
    X_train['duration'].fillna(0, inplace=True)


    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_data_for_modeling(merged_data)


# Initialize the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)

print(X_train.isnull().sum())

# Train the model
lr_model.fit(X_train, y_train)

# Predict on the training data
train_predictions = lr_model.predict(X_train)

# Calculate and print training accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Handle NaN values for the test set
X_test['difficulty'].fillna(0, inplace=True)
X_test['reward'].fillna(0, inplace=True)
X_test['duration'].fillna(0, inplace=True)


# Use the trained logistic regression model to make predictions on the test data
test_predictions = lr_model.predict(X_test)

# Calculate and print test accuracy
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print a classification report to assess precision, recall, and F1-score
print(classification_report(y_test, test_predictions))


