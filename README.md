# Comparative Study Web Application

## Project Overview
This project is a Flask-based web application that allows users to upload a dataset, specify the target column, and perform a comparative study using seven machine learning algorithms. The application provides Exploratory Data Analysis (EDA) and model performance comparisons.

## Features
- Upload CSV datasets
- Perform EDA (summary statistics, missing values, duplicate counts)
- Train and compare multiple machine learning models:
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Gradient Boosting
- View results and analysis in a user-friendly web interface

## Installation

### Prerequisites
Ensure you have Python 3 installed along with the following libraries:

```bash
pip install flask pandas seaborn matplotlib numpy scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/comparative-study.git
cd comparative-study
```

## Running the Application

1. Start the Flask app:
   ```bash
   python comparative_study.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. Upload a CSV file.
2. Enter the target column name (the output variable for prediction).
3. Click "Upload and Analyze" to generate results.
4. View EDA report and model comparison table.

## Project Structure
```
comparative-study/
│-- uploads/               # Uploaded datasets
│-- templates/             # HTML templates
│   ├── index.html          # Upload page
│   ├── report.html         # Results page
│-- static/                 # Static files (CSS, JS)
│   ├── styles.css          # Styling
│   ├── script.js           # Client-side script
│-- comparative_study.py    # Main Flask application
│-- README.md               # Project documentation
```

## Screenshots

1. **Upload Page:**
   - Upload dataset and specify target column.
2. **Results Page:**
   - View EDA insights and model performance.
