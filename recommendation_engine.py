"""
Recommendation Engine for Starbucks Capstone Project

This script contains a class that is designed to build a recommendation engine
for Starbucks. The recommendation engine predicts which offers would likely be 
completed by a given customer based on their demographic information and offer details.

Classes:
    RecommendationEngine: Defines the recommendation engine model and methods 
    for training the model, making predictions, and recommending offers.
"""

import pandas as pd


class RecommendationEngine:
    """
    This class is responsible for building the recommendation engine model.
    
    Attributes:
        model (LogisticRegression): The trained Logistic Regression model.
        X_train (DataFrame): The training data used for the model.
    """
    
    def __init__(self, model, X_train):
        """
        Initialize the RecommendationEngine with a trained model and training data.
        
        Args:
            model (LogisticRegression): The trained Logistic Regression model.
            X_train (DataFrame): The training data used for the model.
        """
        self.model = model
        self.X_train = X_train
    
    def recommend_offer(self, user_data):
        """
        Recommend an offer type for a specific user based on their demographic information.
        
        Args:
            user_data (Series or DataFrame): The demographic information of the user.
            
        Returns:
            int: The recommended offer type (1 for recommended, 0 for not recommended).
        """
        # Check if the input is a pandas Series. If not, convert it to a Series
        if not isinstance(user_data, pd.Series):
            user_data = pd.Series(user_data)
        
        # Make a prediction for this user
        prediction = self.model.predict(user_data.values.reshape(1, -1))
        
        return prediction[0]


if __name__ == '__main__':
    # For testing purposes, one might include code here to instantiate the class
    # and make an example recommendation.
    pass
