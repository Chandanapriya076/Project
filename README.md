# AI-Powered Email Subject Classification

This project focuses on developing an intelligent system that automatically classifies customer support emails into predefined categories. Utilizing machine learning and natural language processing (NLP) techniques, the system analyzes both the subject line and body of incoming emails to accurately predict their most relevant subject category. This automation aims to streamline email routing, improve response times, and enhance overall customer support efficiency by ensuring emails are directed to the appropriate department or handled with the correct priority.

# Problem Statement

The business problem addressed by this project is the inefficient manual routing of customer support tickets, which often leads to delays and misdirection. To solve this, the project aims to develop a machine learning solution that automatically classifies incoming customer support emails. This is framed as a multi-class text classification problem, leveraging both the email's subject line and body content to accurately predict the most appropriate category or team for handling the ticket.

# Solution Approach

To address the multi-class text classification problem, the following approach was implemented:

1.  **Data Preprocessing and Cleaning**: The raw email `body` text underwent a thorough cleaning process using NLTK, which included converting text to lowercase, removing punctuation, eliminating stopwords, and handling extra spaces to prepare the data for feature extraction.
2.  **Feature Extraction with TF-IDF**: Textual data from the cleaned email bodies was transformed into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization technique. This method captures the importance of words within the context of the entire dataset.
3.  **Model Training**: Two distinct machine learning models were trained and compared: a simple Multinomial Naive Bayes (MNB) model, which serves as a robust baseline, and a more powerful ensemble model, the Random Forest Classifier (RFC), to capture complex patterns in the data.
4.  **Model Evaluation**: Both trained models were rigorously evaluated using standard classification metrics, including accuracy score, a comprehensive classification report (precision, recall, F1-score), and confusion matrices, to assess their performance across all subject categories.
5.  **Best Model Selection and Saving**: Based on the evaluation results, the model demonstrating superior performance was selected and saved using `joblib` (as `best_model.pkl`), along with the fitted TF-IDF vectorizer (`tfidf_vectorizer.pkl`), to enable future predictions on new, unseen email data.

# Results

Initial model training and evaluation were conducted on a very small test set. The Multinomial Naive Bayes model achieved an accuracy of approximately 25%. The Random Forest Classifier demonstrated similar performance metrics. Given the limited size of the dataset and the inherent class imbalance, these baseline results indicate the complexity of the task and the potential need for more data, advanced preprocessing, or sophisticated modeling techniques. The best-performing model from this initial phase was saved (`best_model.pkl`) for future prediction tasks.

# How to Run

Follow these steps to set up and run the project:

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed. Then, install the required libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    You may also need to download NLTK data if not already present. The preprocessing script will attempt to do this automatically.

3.  **Run Data Preprocessing and EDA**:
    This script cleans the raw email data, performs exploratory data analysis, and saves the cleaned dataset.
    ```bash
    python 02_data_preprocessing.py
    ```

4.  **Run Model Training**:
    This script loads the cleaned data, trains a baseline (Naive Bayes) and a more powerful (Random Forest) model, evaluates them, and saves the best-performing model along with the TF-IDF vectorizer.
    ```bash
    python 03_model_training.py
    ```

5.  **Run Prediction Script**:
    Use this script to load the trained model and make predictions on new email subjects and bodies.
    ```bash
    python predict.py
    ```

# Dataset Overview

The dataset, named `ai engineer dataset.csv`, contains 20 entries and 4 columns, providing information about customer support emails. Below is a summary of the dataset's structure:

| Column Name | Description | Data Type |
|---|---|---|
| `sender` | The email address of the sender. | Object (String) |
| `subject` | The subject line of the email. | Object (String) |
| `body` | The main content of the email. | Object (String) |
| `sent_date` | The date and time the email was sent. | Object (String) |

## Class Distribution

The 'subject' column, which serves as our target variable for classification, contains 9 unique categories. The distribution of these subjects across the dataset is as follows:

| Subject | Count |
|---|---:|
| Help required with account verification | 4 |
| General query about subscription | 3 |
| Immediate support needed for billing error | 2 |
| Urgent request: system access blocked | 2 |
| Critical help needed for downtime | 2 |
| Question: integration with API | 1 |
| Support needed for login issue | 2 |
| Request for refund process clarification | 1 |
| Query about product pricing | 2 |

This distribution highlights a class imbalance, where some subject categories have significantly more instances than others. This is a common challenge in machine learning, as models trained on imbalanced datasets may become biased towards the majority classes, leading to poor performance in predicting minority classes. Addressing this imbalance through techniques like oversampling, undersampling, or using appropriate evaluation metrics will be crucial for developing a robust classification model.

# Repository Structure

-   `ai engineer dataset.csv`: The raw dataset containing customer support emails.
-   `analyze_csv.py`: A script for initial data loading, basic information display, and unique subject count analysis.
-   `02_data_preprocessing.py`: Handles data cleaning (lowercase, punctuation, stopwords, extra spaces), feature engineering (TF-IDF), and visualization of class distribution.
-   `03_model_training.py`: Contains code for splitting data, training machine learning models (Naive Bayes, Random Forest), evaluating their performance, and saving the best-performing model.
-   `predict.py`: A script to load the trained model and TF-IDF vectorizer, and then predict subject categories for new email inputs.
-   `requirements.txt`: Lists all the Python dependencies required to run the project.
-   `README.md`: Provides an overview of the project, problem statement, solution approach, results, and instructions on how to run the project.
