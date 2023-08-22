"""
Main Module for Starbucks Capstone Challenge

This script serves as the main entry point to the Starbucks Capstone Challenge Project.
It orchestrates the process of data preprocessing, model training, model evaluation,
and offer recommendation for a specific user based on their demographic information.

It performs the following steps:
    1. Preprocesses the raw data to merge and clean the datasets.
    2. Trains a logistic regression model to predict offer completions.
    3. Evaluates the trained model on test data.
    4. Demonstrates how to use the model to recommend an offer type for a specific user.

Example:
    $ python main.py

This will run the entire pipeline and print the recommended offer type for a sample user.

Functions:
    main: The main function to run the entire pipeline.
"""

import pandas as pd
from data_preparation import preprocess_data
from ml_modeling import train_model, evaluate_model
from recommendation_engine import RecommendationEngine
from data_preparation import load_data

def main():
    """
    Main function to run the entire pipeline.
    It preprocesses the data, trains a model, evaluates it, and demonstrates a recommendation.
    """
    # Load the datasets
    portfolio, profile, transcript = load_data()
    
    # Step 1: Preprocess the data
    merged_data = preprocess_data(portfolio, profile, transcript)
    
    # Step 2: Train the model
    lr_model, X_test, y_test, X_train = train_model(merged_data)
    
    # Step 3: Evaluate the model
    evaluate_model(lr_model, X_test, y_test)
    
    # Step 4: Demonstrate a recommendation
    rec_engine = RecommendationEngine(lr_model, X_train)
    user_data = pd.Series({
        'age': 35,
        'income': 80000,
        'gender_F': 1,
        'gender_M': 0,
        'gender_O': 0,
        'offer_bogo': 1,
        'offer_discount': 0,
        'offer_informational': 0,
        'difficulty': 5,
        'reward': 5,
        'duration': 5
    })
    offer_recommendation = rec_engine.recommend_offer(user_data)
    print("Recommended Offer Type:", offer_recommendation)


if __name__ == '__main__':
    main()
