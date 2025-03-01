# ==============================================================================
# Filename: automl.py
# ==============================================================================
# Purpose:
# ==============================================================================
# This script automates the entire machine learning pipeline from data loading 
# to model prediction. It performs key operations such as data preprocessing, 
# model selection, model evaluation, hyperparameter tuning, and final model 
# training. The script supports both classification and regression tasks, 
# and is designed to handle numerical, categorical, and text data.
# 
# ==============================================================================
# Key Functionalities:
# ==============================================================================
# 1. Data Loading and Preprocessing:
#    - Loads datasets in CSV format.
#    - Automatically detects and processes different types of data, including 
#      numerical, categorical, and text features.
#    - Handles missing data using imputation techniques.
#    - Scales numerical features using StandardScaler.
#    - Encodes categorical variables using OneHotEncoder.
#    - Text features are transformed into numerical representations using 
#      TF-IDF (Term Frequency-Inverse Document Frequency).
#    
# 2. Model Selection and Evaluation:
#    - Provides a set of machine learning models for both classification and 
#      regression tasks, including the following:
#    
#    Classification Models:
#    - Logistic Regression
#    - Random Forest Classifier
#    - Support Vector Machine (SVM)
#    - Decision Tree Classifier
#    - K-Nearest Neighbors (KNN) Classifier
#    - Gradient Boosting Classifier
#    - AdaBoost Classifier
#    - LightGBM Classifier
#    - CatBoost Classifier
#    - Ridge Classifier
#    
#    Regression Models:
#    - Linear Regression
#    - Random Forest Regressor
#    - Support Vector Regressor (SVR)
#    - Decision Tree Regressor
#    - K-Nearest Neighbors (KNN) Regressor
#    - Gradient Boosting Regressor
#    - AdaBoost Regressor
#    - LightGBM Regressor
#    - CatBoost Regressor
#    - Ridge Regressor
#    
#    - Models are evaluated using cross-validation to determine the best performing 
#      models based on accuracy (for classification) or mean squared error (for 
#      regression).
#    
# 3. Hyperparameter Tuning using GridSearchCV:
#    - Uses GridSearchCV to tune hyperparameters for the selected model, optimizing 
#      them based on cross-validation scores.
#    - Allows fine-tuning of models such as Random Forest, SVM, Gradient Boosting, 
#      and more.
#    
# 4. Final Model Training and Evaluation on Test Data:
#    - The best performing model is trained on the entire dataset (training + testing data).
#    - Model performance is evaluated on the test data using appropriate 
#      metrics (accuracy for classification, MSE for regression).
#    
# 5. Saving the Trained Model for Future Use:
#    - Once the model is trained and evaluated, it is saved to disk using the 
#      `joblib` library to facilitate future use or deployment without needing to 
#      retrain the model.
#    
# 6. Model Prediction using User Input:
#    - Allows for real-time prediction by accepting user input.
#    - Preprocesses the input data (imputation, scaling, encoding) and predicts 
#      the output using the trained model.
# ==============================================================================


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
import catboost as cb
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class TextColumnProcessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to handle missing data imputation and text vectorization (TF-IDF).
    
    Parameters:
    vectorizer : TfidfVectorizer, optional (default=None)
        The vectorizer to be used for transforming text data. If None, the default TF-IDF vectorizer is used.
    
    Attributes:
    imputer : SimpleImputer
        Imputer used to fill missing values with a constant value 'missing'.
    vectorizer : TfidfVectorizer
        The vectorizer for transforming text data into numerical form using TF-IDF.
    """
    
    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer()
        self.imputer = SimpleImputer(strategy='constant', fill_value='missing')

    def fit(self, X, y=None):
        """
        Fit the imputer and vectorizer on the data.

        Parameters:
        X : pd.Series or np.ndarray
            The input data (single column text data).

        y : None, optional (default=None)
            Not used, present for compatibility with scikit-learn.

        Returns:
        self : object
            The fitted transformer.
        """
        self.imputer.fit(X)
        text_data = self.imputer.transform(X).astype(str).flatten()
        self.vectorizer.fit(text_data)
        return self

    def transform(self, X):
        """
        Transform the data by imputing missing values and applying TF-IDF vectorization.

        Parameters:
        X : pd.Series or np.ndarray
            The input data (single column text data).

        Returns:
        np.ndarray
            The transformed text data in numerical form.
        """
        text_data = self.imputer.transform(X).astype(str).flatten()
        return self.vectorizer.transform(text_data).toarray()


def convert_to_1d_if_single_column(X):
    """
    Convert the input data to a 1D array if the input has only one column.
    
    Parameters:
    X : pd.DataFrame or np.ndarray
        The input data.
        
    Returns:
    np.ndarray
        The transformed 1D array if the input has a single column.
    """
    if X.ndim == 1:
        X = X.values.ravel()
    return X


def check_shape(X, y):
    """
    Check and ensure that the features and target datasets have the correct shape.
    
    Parameters:
    X : pd.DataFrame or np.ndarray
        The features dataset.
        
    y : pd.Series or np.ndarray
        The target dataset.
    """
    assert X.shape[0] == y.shape[0], "Features and target datasets must have the same number of rows."


def load_data(file_path):
    """
    Load the dataset from the specified file path.

    Parameters:
    file_path : str
        The path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame
        The loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


def identify_problem_type(data, target_column):
    """
    Identify the type of machine learning problem (classification or regression).

    Parameters:
    data : pd.DataFrame
        The input dataset.
        
    target_column : str
        The name of the target column.

    Returns:
    str
        The problem type: 'classification' or 'regression'.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    target_data = data[target_column]
    if target_data.dtype == 'object' or target_data.nunique() <= 10:
        return 'classification'
    else:
        return 'regression'


def identify_target_features(data, target_column):
    """
    Identify and separate the target column from the feature columns.

    Parameters:
    data : pd.DataFrame
        The input dataset.
        
    target_column : str
        The name of the target column.

    Returns:
    X : pd.DataFrame
        The features dataset.
        
    y : pd.Series
        The target dataset.
        
    problem_type : str
        The type of the problem ('classification' or 'regression').
    """
    problem_type = identify_problem_type(data, target_column)
    X = data.drop(columns=[target_column]).copy()
    y = data[target_column].copy()
    return X, y, problem_type


def build_preprocessing_pipeline(X):
    """
    Build a preprocessing pipeline for numerical, categorical, and text data.

    Parameters:
    X : pd.DataFrame
        The features dataset.

    Returns:
    ColumnTransformer
        The preprocessing pipeline that applies different transformations to different feature types.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X_copy = X.copy()
    numerical_cols = X_copy.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_copy.select_dtypes(include=['object']).columns.tolist()
    
    text_cols = [col for col in categorical_cols if X_copy[col].apply(lambda x: isinstance(x, str) and len(x.split()) > 2).any()]

    numerical_pipeline = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    text_pipeline = Pipeline(steps=[ 
        ('text_processor', TextColumnProcessor())
    ])

    categorical_pipeline = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[ 
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, [col for col in categorical_cols if col not in text_cols]),
            ('text', text_pipeline, text_cols)
        ])


def get_models(problem_type):
    """
    Retrieve a list of models for classification or regression tasks.

    Parameters:
    problem_type : str
        The type of the problem ('classification' or 'regression').

    Returns:
    list
        A list of tuples where each tuple contains the model name and the model instance.
    """
    if problem_type == 'classification':
        return [
            ('Logistic Regression', LogisticRegression(max_iter=10000)),
            ('Random Forest', RandomForestClassifier(verbose=0)),
            ('SVM', SVC(verbose=False)),
            ('Decision Tree', DecisionTreeClassifier()),
            ('KNN', KNeighborsClassifier()),
            ('Gradient Boosting', GradientBoostingClassifier(verbose=0)),
            ('AdaBoost', AdaBoostClassifier()),
            ('LightGBM', lgb.LGBMClassifier(verbose=-1)),
            ('CatBoost', cb.CatBoostClassifier(verbose=0)),
            ('Ridge Classifier', RidgeClassifier())
        ]
    else:
        return [
            ('Linear Regression', LinearRegression()),
            ('Random Forest Regressor', RandomForestRegressor(verbose=0)),
            ('SVR', SVR(verbose=False)),
            ('Decision Tree Regressor', DecisionTreeRegressor()),
            ('KNN Regressor', KNeighborsRegressor()),
            ('Gradient Boosting Regressor', GradientBoostingRegressor(verbose=0)),
            ('AdaBoost Regressor', AdaBoostRegressor()),
            ('LightGBM Regressor', lgb.LGBMRegressor(verbose=-1)),
            ('CatBoost Regressor', cb.CatBoostRegressor(verbose=0)),
            ('Ridge Regressor', Ridge())
        ]


def evaluate_models(X_train, y_train, problem_type):
    """
    Evaluate different models using cross-validation.

    Parameters:
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
        
    y_train : pd.Series or np.ndarray
        The training target data.
        
    problem_type : str
        The type of the problem ('classification' or 'regression').

    Returns:
    list
        A sorted list of tuples where each tuple contains the model name and its average score.
    """
    models = get_models(problem_type)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if problem_type == 'classification' else 5
    results = []

    for name, model in models:
        pipeline = Pipeline(steps=[('preprocessor', build_preprocessing_pipeline(X_train)), ('model', model)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy' if problem_type == 'classification' else 'neg_mean_squared_error')
        results.append((name, np.mean(scores)))
    
    return sorted(results, key=lambda x: x[1], reverse=True)


def tune_hyperparameters(model_name, model, X_train, y_train, problem_type):
    """
    Tune hyperparameters for a specific model using GridSearchCV.

    Parameters:
    model_name : str
        The name of the model to tune.
        
    model : estimator object
        The model to be tuned.
        
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
        
    y_train : pd.Series or np.ndarray
        The training target data.
        
    problem_type : str
        The type of the problem ('classification' or 'regression').

    Returns:
    estimator object
        The model with the best hyperparameters after tuning.
    """
    param_grid = {
        'Logistic Regression': {'model__C': [0.1, 1, 10]},
        'Random Forest': {'model__n_estimators': [50, 100, 250], 'model__max_depth': [5, 10, 25, 50]},
        'SVM': {'model__C': [0.1, 1], 'model__kernel': ['linear', 'rbf']},
        'Decision Tree': {'model__max_depth': [5, 10, 20]},
        'KNN': {'model__n_neighbors': [3, 5, 7]},
        'Gradient Boosting': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'AdaBoost': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'XGBoost': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'LightGBM': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'CatBoost': {'model__iterations': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'Ridge Classifier': {'model__alpha': [0.1, 1, 10]},
        'Linear Regression': {'model__fit_intercept': [True, False]},
        'SVR': {'model__C': [0.1, 1], 'model__kernel': ['linear', 'rbf']},
        'Random Forest Regressor': {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]},
        'Decision Tree Regressor': {'model__max_depth': [5, 10, 20]},
        'KNN Regressor': {'model__n_neighbors': [3, 5, 7]},
        'Gradient Boosting Regressor': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'AdaBoost Regressor': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'XGBoost Regressor': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'LightGBM Regressor': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'CatBoost Regressor': {'model__iterations': [50, 100], 'model__learning_rate': [0.01, 0.1]},
        'Ridge Regressor': {'model__alpha': [0.1, 1, 10]},
    }

    if model_name in param_grid:
        grid_search = GridSearchCV(Pipeline(steps=[('preprocessor', build_preprocessing_pipeline(X_train)), ('model', model)]),
                                   param_grid=param_grid[model_name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    return model


def train_and_evaluate_best_model(X_train, y_train, X_test, y_test, best_model_name, problem_type):
    """
    Train and evaluate the best model on the training and test sets.

    Parameters:
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
        
    y_train : pd.Series or np.ndarray
        The training target data.
        
    X_test : pd.DataFrame or np.ndarray
        The test feature data.
        
    y_test : pd.Series or np.ndarray
        The test target data.
        
    best_model_name : str
        The name of the best model.
        
    problem_type : str
        The type of the problem ('classification' or 'regression').

    Returns:
    Pipeline
        The trained pipeline (including the model and preprocessing steps).
    """
    models = get_models(problem_type)
    best_model = [model for name, model in models if name == best_model_name][0]
    
    pipeline = Pipeline(steps=[('preprocessor', build_preprocessing_pipeline(X_train)), ('model', best_model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    if problem_type == 'classification':
        score = accuracy_score(y_test, y_pred)
    else:
        score = mean_squared_error(y_test, y_pred)
    
    return pipeline


def save_model(best_model):
    """
    Save the trained model to a file.

    Parameters:
    best_model : Pipeline
        The trained pipeline to be saved.
    """
    joblib.dump(best_model, 'best_model.pkl')


def predict(model):
    """
    Predict the output using the trained model based on user input.

    Parameters:
    model : Pipeline
        The trained pipeline used for prediction.

    Returns:
    np.ndarray
        The prediction made by the model based on the user's input.
    """
    preprocessor = model.named_steps['preprocessor']
    
    numerical_cols = preprocessor.transformers_[0][2]
    categorical_cols = preprocessor.transformers_[1][2]
    text_cols = preprocessor.transformers_[2][2]

    user_input = {}

    all_columns = numerical_cols + categorical_cols + text_cols
    
    for col in all_columns:
        value = input(f"Enter value for '{col}': ").strip()
        
        if col in numerical_cols:
            try:
                user_input[col] = float(value)
            except ValueError:
                user_input[col] = 0.0
        else:
            user_input[col] = value

    user_input_df = pd.DataFrame([user_input])
    prediction = model.predict(user_input_df)

    return prediction[0]


def train_model(file_path, target_column):
    """
    Train and evaluate a machine learning model based on the dataset provided.

    Parameters:
    file_path : str
        The path to the dataset CSV file.
        
    target_column : str
        The name of the target column.

    Returns:
    Pipeline
        The trained pipeline containing the best model.
    """
    data = load_data(file_path)
    X, y, problem_type = identify_target_features(data, target_column)

    check_shape(X, y)
    X = convert_to_1d_if_single_column(X)
    check_shape(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy(), test_size=0.2, random_state=42)

    results = evaluate_models(X_train, y_train, problem_type)
    best_model_name = results[0][0]

    best_model = tune_hyperparameters(best_model_name, [model for name, model in get_models(problem_type) if name == best_model_name][0], X_train, y_train, problem_type)

    final_model = train_and_evaluate_best_model(X_train, y_train, X_test, y_test, best_model_name, problem_type)
    save_model(final_model)

    return final_model
