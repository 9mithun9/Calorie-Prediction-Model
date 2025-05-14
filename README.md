Calorie Expenditure Prediction Model

Project Overview

This project develops a robust ensemble machine learning model to predict calorie expenditure based on factors such as age, height, weight, heart rate, and exercise duration. Utilizing Random Forest and XGBoost regressors, the model incorporates feature engineering (e.g., BMI, ACSM formula) and addresses predictive accuracy through ensembling. The project showcases skills in data preprocessing, model training, evaluation, and deployment, aligning with the requirements of an Artificial Intelligence Development Specialist role.

Key Features





Data Preprocessing: Handles numerical features, encodes binary gender, and computes BMI and ACSM calorie estimates.



Feature Engineering: Adds BMI and ACSM formula-based features to enhance model performance.



Model: Ensembles Random Forest and XGBoost regressors with weighted predictions.



Evaluation: Uses RMSLE (Root Mean Squared Logarithmic Error) as the evaluation metric, per Kaggle competition rules.



Output: Generates a submission file (submission_ensemble.csv) with predictions on test data.

Technologies Used





Python: Core programming language.



Pandas & NumPy: For data manipulation and numerical operations.



Scikit-learn: For Random Forest model and evaluation metrics.



XGBoost: For gradient boosting model.



Opendatasets: For downloading Kaggle competition data.



Matplotlib & Seaborn: For potential visualization (extendable).

Dataset

The dataset is sourced from the Kaggle competition "Playground Series S5E5" (https://www.kaggle.com/competitions/playground-series-s5e5/data). It includes:





Features: Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Sex (male/female), and derived features (BMI, ACSM).



Target: Calories (continuous variable for regression).



Files: train.csv, test.csv, sample_submission.csv.

Installation





Clone the repository:

git clone https://github.com/your-username/Calorie-Expenditure-Prediction.git
cd Calorie-Expenditure-Prediction



Install dependencies:

pip install -r requirements.txt

The requirements.txt should include:

pandas
numpy
scikit-learn
xgboost
opendatasets
matplotlib
seaborn



Install opendatasets (if not included in requirements):

pip install opendatasets

Dataset Download





Download the Kaggle dataset:





Run the following Python code to download the dataset from the Kaggle competition:

import opendatasets as od
dataset_url = 'https://www.kaggle.com/competitions/playground-series-s5e5/data'
od.download(dataset_url)



When prompted, provide your Kaggle username and API key. Learn more about Kaggle credentials: http://bit.ly/kaggle-creds.



The dataset will be extracted to the ./playground-series-s5e5 directory.



Set data directory:





Ensure the dataset files (train.csv, test.csv, sample_submission.csv) are in the playground-series-s5e5 folder.



Update the script to point to the correct data directory:

data_dir = 'playground-series-s5e5'

Usage





Run the script:

python calorie_expenditure_model.py

Note: Ensure calorie_expenditure_model.py contains the provided code (data loading, feature engineering, model training, and ensembling).



Outputs:





Console Output: Training and validation RMSLE scores for Random Forest, XGBoost, and the ensemble model.





Example: Ensemble Validation RMSLE: 0.06099



Submission File: submission_ensemble.csv with predictions for the test set.



Visualizations (optional): Add plots (e.g., feature importance) by extending the script with Matplotlib/Seaborn.

Project Structure

Calorie-Expenditure-Prediction/
├── calorie_expenditure_model.py  # Main script
├── playground-series-s5e5/      # Dataset directory
│   ├── train.csv                # Training data
│   ├── test.csv                 # Test data
│   ├── sample_submission.csv    # Sample submission
├── submission_ensemble.csv      # Output: Submission file
├── requirements.txt             # Dependencies
└── README.md                    # This file

Feature Engineering





Binary Encoding: Converted Sex to binary (male: 1, female: 0).



BMI Calculation: Computed as Weight / (Height/100)^2.



ACSM Formula: Added calorie estimates using the ACSM formula, adjusted for gender, heart rate, weight, age, and duration.

Model Details





Random Forest Regressor:





Parameters: max_depth=17, min_samples_split=15, n_estimators=400.



Performance: Train RMSLE: 0.04911, Val RMSLE: 0.06160.



XGBoost Regressor:





Parameters: n_estimators=700, learning_rate=0.08, max_depth=6.



Performance: Train RMSLE: 0.05917, Val RMSLE: 0.06298.



Ensemble:





Weighted average (60% Random Forest, 40% XGBoost).



Performance: Ensemble Validation RMSLE: 0.06099.

Results





The ensemble model achieves a validation RMSLE of 0.06099, indicating strong predictive performance on unseen data.



The submission file (submission_ensemble.csv) is formatted for Kaggle competition submission, predicting calorie expenditure for the test set.

Future Improvements





Incorporate additional features (e.g., interaction terms or polynomial features).



Experiment with advanced ensembling techniques (e.g., stacking).



Add visualizations for feature importance and prediction errors.



Deploy the model as a web application using Flask or FastAPI.

Author

MD Mehedi Hasan Mithun 
https://www.linkedin.com/in/md-mehedi-hasan-mithun-1428b1124/
9mithun9@gmail.com 
