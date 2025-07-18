# Employee-salary-prediction
Employee Income Classification: Predicting Income Brackets
Project Overview
This project aims to develop a machine learning model to classify an individual's income into two categories: >50K (greater than $50,000 per year) or <=50K (less than or equal to $50,000 per year). This classification task is essential for various applications, including targeted marketing, resource allocation, and understanding socio-economic patterns. The project utilizes the Adult Census Income dataset to build a robust predictive model.

Table of Contents
Problem Statement

Dataset

System Requirements

Libraries Used

Algorithm & Methodology

How to Run the Project

Results

Conclusion

Future Scope

References

Problem Statement
The core problem addressed is the accurate prediction of an individual's income bracket based on various demographic and employment-related features. This involves handling complex relationships within the data, managing missing values, and effectively transforming raw features into a format suitable for machine learning models. The goal is to provide a reliable classification model that can be used for informed decision-making.

Dataset
The project uses the adult 3.csv dataset, which is a modified version of the Adult Census Income dataset. It contains various attributes about individuals extracted from the 1994 Census database.

Key features include:

age: Age of the individual.

workclass: Type of employer.

education: Highest level of education achieved.

education-num: Numeric representation of education level.

marital-status: Marital status.

occupation: Type of occupation.

relationship: Relationship status.

race: Race of the individual.

sex: Gender.

capital-gain: Capital gains.

capital-loss: Capital losses.

hours-per-week: Number of hours worked per week.

native-country: Country of origin.

income: Target variable, either >50K or <=50K.

System Requirements
To run this project, you will need:

Operating System: Windows, macOS, or Linux

Programming Language: Python 3.x

Hardware: A standard computer with sufficient RAM (e.g., 8GB or more is recommended for data processing).

Libraries Used
The following Python libraries are essential and can be installed via pip:

pandas: For data manipulation and analysis.

scikit-learn (sklearn): For machine learning functionalities, including:

model_selection: train_test_split

preprocessing: StandardScaler, OneHotEncoder

compose: ColumnTransformer

pipeline: Pipeline

ensemble: RandomForestClassifier

metrics: accuracy_score, classification_report, confusion_matrix

impute: SimpleImputer

numpy: For numerical operations.

You can install these dependencies using the following command:

pip install pandas scikit-learn numpy

Algorithm & Methodology
The project employs a robust machine learning pipeline to classify income, centered around a Random Forest Classifier. The methodology involves the following key steps:

Data Loading and Initial Inspection:

The adult 3.csv dataset is loaded, with column names explicitly defined.

Initial data insights are gathered using df.head() and df.info().

Missing values, represented as '?', are identified and converted to np.nan.

Data Cleaning and Preprocessing:

The 'fnlwgt' column is dropped as it is a sampling weight and not a predictive feature.

Features are categorized into numerical and categorical types.

The income column is extracted and removed from the feature set.

Target Variable Preparation:

The income column is transformed into a binary target variable, income_over_50k, where >50K maps to 1 and <=50K maps to 0.

Feature Engineering and Preprocessing Pipeline:

Numerical Features: Missing values are imputed using the median strategy (SimpleImputer), and features are then scaled using StandardScaler to normalize their ranges.

Categorical Features: Missing values are imputed using the most frequent strategy (SimpleImputer), and then features are converted into numerical format using OneHotEncoder (handle_unknown='ignore' is used for robustness).

A ColumnTransformer orchestrates the application of these specific transformers to their respective feature types.

Model Training:

The dataset is split into training (80%) and testing (20%) sets using train_test_split, ensuring stratification of the target variable for balanced class representation.

A RandomForestClassifier with 100 estimators (n_estimators=100) is chosen as the classification model due to its ensemble nature and strong performance.

A Pipeline combines the preprocessing steps and the classifier, streamlining the entire model training process.

The model is trained on the prepared training data.

Model Evaluation:

The trained model's performance is assessed on the unseen test set using:

Accuracy Score: Overall proportion of correct predictions.

Classification Report: Detailed metrics including precision, recall, and F1-score for each class.

Confusion Matrix: A table summarizing true positives, true negatives, false positives, and false negatives.

How to Run the Project
Download the Code and Data:

Save the Python script (employee_income_classification_project.py) to your local machine.

Ensure you have the adult 3.csv dataset in the same directory as your Python script.

Set up Python Environment:

Make sure you have Python 3.x installed.

Install the required libraries using pip:

pip install pandas scikit-learn numpy

Run the Script:

Open your terminal or command prompt.

Navigate to the directory where you saved your project files.

Execute the Python script:

python employee_income_classification_project.py

Running in Google Colab
Open a new Google Colab notebook (colab.research.google.com).

Upload adult 3.csv to the Colab session storage (Files icon -> Upload button).

(Optional) Run !pip install pandas scikit-learn numpy in a code cell to ensure libraries are installed.

Paste the contents of employee_income_classification_project.py into a code cell.

Run the cell.

Results
Upon running the script, you will see detailed output in your console, including:

Confirmation of dataset loading and initial rows.

Dataset information and a summary of missing values.

Identification of categorical and numerical features.

Target variable distribution.

Confirmation of data splitting and model training completion.

The final evaluation metrics:

Accuracy: Approximately 0.8572 (85.72%)

Classification Report:

              precision    recall  f1-score   support

           0       0.88      0.94      0.91      4944
           1       0.75      0.59      0.66      1569

    accuracy                           0.86      6513
   macro avg       0.82      0.77      0.78      6513
weighted avg       0.85      0.86      0.85      6513

Confusion Matrix:

[[4638  306]
 [ 624  945]]

(Note: Actual numerical results may vary slightly based on environment and library versions, but will be very close to these values.)

Conclusion
The project successfully built a machine learning model to classify employee income brackets. The Random Forest Classifier, combined with a robust preprocessing pipeline, demonstrated strong predictive capabilities with an accuracy of around 85.7%. The model effectively handles missing data and categorical features, providing a reliable solution for income classification. The evaluation metrics highlight good performance for both income classes, with a slightly better performance on the majority class (<=50K).

Future Scope
Hyperparameter Tuning: Implement advanced tuning techniques (e.g., GridSearchCV, RandomizedSearchCV) to optimize model performance.

Advanced Feature Engineering: Explore creating more complex interaction features or polynomial features.

Ensemble Methods: Experiment with other powerful ensemble techniques like Gradient Boosting (XGBoost, LightGBM) or Stacking for potential accuracy improvements.

Addressing Class Imbalance: Apply techniques such as SMOTE or adjust class weights to further improve the model's ability to predict the minority class (>50K).

Model Interpretability: Utilize tools like SHAP or LIME to gain deeper insights into individual predictions and overall feature contributions.

Deployment: Develop a simple web application (e.g., using Flask or FastAPI) to serve the trained model for real-time predictions.

References
Dua, D. and Graff, C. (2017). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science. (Adult dataset)

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

