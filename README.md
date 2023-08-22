# Starbucks Capstone Challenge Project

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [In-Depth Data Analysis](#in-depth-data-analysis)
- [Recommendation Engine](#recommendation-engine)
- [Methodology](#methodology)
- [Results and Model Performance](#results-and-model-performance)
- [How To Use This Project](#how-to-use-this-project)
- [Challenges and Future Work](#challenges-and-future-work)
- [Acknowledgements](#acknowledgements)

## Project Overview

### Domain
This project falls under the domain of customer segmentation and recommendation systems in the context of retail and marketing. Starbucks, one of the world’s most popular coffee shop chains, regularly sends offers to its customers as part of its marketing strategy. The effectiveness of these offers can vary significantly among different customer demographics. Understanding which offers are most likely to be completed by different types of customers is a critical task for optimizing marketing strategies and enhancing customer engagement.

### Origin of the Project
This project is designed as a capstone challenge for the Udacity Data Science Nanodegree Program. The primary goal of the project is to build a demographic-based offer recommendation system for Starbucks. This system aims to predict which offers are likely to be completed by different demographic groups, based on historical data provided by Starbucks. The aim is to leverage machine learning techniques to make these predictions, thereby enabling more personalized and effective marketing strategies.

### Datasets
The data for this project is provided by Starbucks and consists of three main files:

1. **`portfolio.json`**: This file contains metadata about each offer that Starbucks sends out. It includes attributes such as the offer’s type, difficulty level, reward, and duration.
2. **`profile.json`**: This dataset contains demographic data for each customer, including age, gender, income, and the date they became a Starbucks rewards member.
3. **`transcript.json`**: This file contains records for transactions, offers received, offers viewed, and offers completed. It provides a detailed log of customer behavior in response to the offers they received.

## Installation

- Python 3.7+
- Pandas
- NumPy
- scikit-learn
- Jupyter Notebook

## File Descriptions

- `data_preparation.py`: Contains functions for loading, cleaning, and merging the datasets, and segmenting users based on age.
- `ml_modeling.py`: Contains functions and procedures to train a logistic regression machine learning model that predicts whether a customer will complete an offer based on their demographics and offer details.
- `recommendation_engine.py`: This file defines the Recommendation Engine, which uses the trained machine learning model to suggest the most appropriate offer type for a specific user based on their demographic information.
- `main.py`: The main entry point to the project. It orchestrates the process of data preprocessing, model training, model evaluation, and offer recommendation for a specific user based on their demographic information.
- `Data Analysis.ipynb`: This Jupyter notebook file contains in-depth data analysis and visualization for understanding customer behavior and offer effectiveness.

## In-Depth Data Analysis

For analysis purposes, we have created a Jupyter notebook file called 'Data Analysis.ipynb'. This notebook explores various aspects of the Starbucks data to gain insights into customer behavior, offer types, and the effectiveness of these offers.

## Recommendation Engine (`recommendation_engine.py`)

1. **Initialization**: 
   - Instantiate the `RecommendationEngine` with the trained machine learning model and a dataset containing user demographic information and offer details.

2. **Training the Recommendation Engine**: 
   - The engine uses the provided trained machine learning model to make predictions. 

3. **Generate Personalized Offer Recommendations**: 
   - Input a user's demographic information (age, income, gender) into the recommendation engine. The engine will use the trained model to predict the likelihood of this user completing various types of offers.
   - The engine will recommend the offer type that the user is most likely to complete.

## Methodology

- Data Loading and Exploration
- Data Preprocessing
- Machine Learning Modeling
- Offer Recommendation

## Results and Model Performance

The Logistic Regression model achieved the following performance:

- Training Accuracy: Approximately 86.34%
- Test Accuracy: Approximately 86.18%

## How To Use This Project

This section provides a step-by-step guide on how to train the machine learning model and generate personalized offer recommendations for Starbucks customers using this project.

### Step 1: Data Preparation
Before you begin, you need to prepare the data.

- **Action**: Run the `data_preparation.py` script to generate the cleaned and merged dataset.

### Step 2: Train the Machine Learning Model
Once the data is prepared, the next step is to train a machine learning model, which in this case is a logistic regression model.

- **Action**: Run the `ml_modeling.py` script to train and evaluate the logistic regression model.

### Step 3: Use the Recommendation Engine
With a trained machine learning model in hand, you can now use that model to make personalized offer recommendations for individual Starbucks customers.

- **Action**:
  1. Load the trained machine learning model from Step 2.
  2. Instantiate the `RecommendationEngine` class from `recommendation_engine.py` with the loaded model.

### Step 4: Deploy and Use in a Real-world Scenario
At this stage, you can integrate this system into Starbucks' existing customer engagement pipelines.

### Step 5: Monitor and Iterate


Regularly update the machine learning model with new data and continue to monitor its performance to ensure it remains effective over time.

## Challenges and Future Work

- Data Imbalance: One of the main challenges encountered during this project was dealing with class imbalance in the dataset.
- Scalability: The current system is not designed to scale easily with the increase in the number of offers or user demographics.

## Acknowledgements

I would like to thank Starbucks for providing the data for this capstone project and Udacity for creating an enriching Data Science Nanodegree Program that made this project possible.