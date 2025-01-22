from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import os

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def perform_eda(df):
    eda_report = {
        "info": str(df.info()),
        "describe": df.describe(include='all').to_html(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum()
    }
    return eda_report

def encode_categorical(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def train_models(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    return results

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        target_column = request.form['target_column']
        if file.filename != '' and target_column:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            df = load_dataset(file_path)
            eda_report = perform_eda(df)
            df_encoded = encode_categorical(df)
            results = train_models(df_encoded, target_column)
            return render_template('report.html', eda_report=eda_report, results=results, target_column=target_column)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)