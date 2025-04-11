Disease-Gene Prediction: A Machine Learning Perspective
Overview
This project explores the association between diseases and genes using machine learning techniques. A web-based application built with Flask allows researchers and lab technicians to input genetic data and receive predictions about possible disease associations.

Dataset
The dataset used in this project is a TSV file containing genetic and disease-related data. It was collected from an online source and preprocessed to ensure optimal performance of machine learning models.

Technologies Used
Frontend: HTML, CSS, JavaScript
Backend: Python (Flask)
Machine Learning Models:
Novel XGBoost
Novel Random Forest
K-Nearest Neighbors (KNN)
LightGBM
Database: MySQL (mysql.connector) for storing user credentials
Data Preprocessing
To prepare the dataset for training, the following preprocessing steps were applied:

Handling missing values
Label encoding
Normalization
KMeans clustering
Train-test split
Model Training and Evaluation
After preprocessing, the dataset was used to train four machine learning models:

Novel XGBoost
Novel Random Forest
KNN
LightGBM
Among these models, Novel Random Forest achieved the highest accuracy of 97.8%, closely followed by Novel XGBoost. Therefore, the Novel Random Forest model was chosen for deployment using joblib.

Application Workflow
The user (lab technician or researcher) enters genetic data into the web-based application.
The backend (Flask) processes the input and feeds it to the trained Novel Random Forest model.
The model predicts whether the input data corresponds to:
The exact disease (Class 0)
A disease group (Class 1)
A disease phenotype (Class 2)
The results are displayed to the user in an interpretable format.
Large Files and Additional Resources
Due to GitHub file size limitations, some large files are stored on Google Drive. You can access them here:

Conclusion
This project provides a machine learning-based approach to identifying disease-gene associations, aiding researchers in genetic analysis and disease prediction. By leveraging advanced models, particularly Novel Random Forest, this application achieves high accuracy and facilitates informed decision-making in medical research.

