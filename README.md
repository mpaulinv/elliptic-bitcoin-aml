# elliptic-bitcoin-aml
Repository for the cryptocurrency AML detection project
Please execute the files in the following order:
1.- requirements.txt 
2.- download_data.py - simple script to download the Elliptic dataset from Kaggle. Make sure to have Kaggle API installed and configured
3.- exploratory_data_analysis.py - This script performs exploratory data analysis (EDA) on the Elliptic Bitcoin dataset. It includes loading the dataset, checking for missing values, visualizing distributions, and analyzing correlations.
4.- feature_engineering.py This script creates features for the prediction of ilicit activity in the elliptic dataset. The features are based on the graph structure of the transactions 
5.- anomaly_detection.py This script will conduct the analysis of the association between the features created in the feature engineering script and available in the dataset, and the target variable.
6.- model_selection.py This script will adjust the proposed models to the data and evaluate their performance.
7.- final_model.py Assessment of the final model performance on the test set. Visualization of the features relationship with the target. Visualization of the graphs comparing the model prediction with the actual labels. 
