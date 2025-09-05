# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
cleaned_data_path = 'cleaned_dataset.csv'
model_output_path = 'best_model.pkl'

# 1. Load the data
df = pd.read_csv(cleaned_data_path)
print("Cleaned dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
print("First 5 rows of the cleaned DataFrame:")
print(df.head())
print("\n")

# Prepare features (X) and target (y)
X = df['cleaned_body']
y = df['subject']

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")
print("\n")

# 3. Use TF-IDF for feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF feature extraction completed. Number of features: {X_train_tfidf.shape[1]}")
print("\n")

# Store model performance for comparison
model_performance = {}

# 4. Train a simple baseline model (Multinomial Naive Bayes)
print("Training Multinomial Naive Bayes model...")
mnb_model = MultinomialNB()
mnb_model.fit(X_train_tfidf, y_train)
y_pred_mnb = mnb_model.predict(X_test_tfidf)
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
model_performance['MultinomialNB'] = accuracy_mnb
print(f"Multinomial Naive Bayes Accuracy: {accuracy_mnb:.4f}")
print("Multinomial Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_mnb, zero_division=0))

# Confusion Matrix for MNB
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_mnb), annot=True, fmt='d', cmap='Blues',
            xticklabels=mnb_model.classes_, yticklabels=mnb_model.classes_)
plt.title('Confusion Matrix - Multinomial Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_mnb.png')
plt.close()  # CHANGED: Close the plot instead of showing it
print("Confusion Matrix for Multinomial Naive Bayes saved as 'confusion_matrix_mnb.png'.")
print("\n")

# 5. Train a more powerful model (Random Forest Classifier)
print("Training Random Forest Classifier model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
model_performance['RandomForestClassifier'] = accuracy_rf
print(f"Random Forest Classifier Accuracy: {accuracy_rf:.4f}")
print("Random Forest Classifier Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# Confusion Matrix for RF
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens',
            xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png')
plt.close()  # CHANGED: Close the plot instead of showing it
print("Confusion Matrix for Random Forest Classifier saved as 'confusion_matrix_rf.png'.")
print("\n")

# 6. Save the best model
best_model_name = max(model_performance, key=model_performance.get)
best_accuracy = model_performance[best_model_name]

if best_model_name == 'MultinomialNB':
    best_model = mnb_model
else:
    best_model = rf_model

joblib.dump(best_model, model_output_path)
print(f"Best model ({best_model_name}) with accuracy {best_accuracy:.4f} saved to {model_output_path}")