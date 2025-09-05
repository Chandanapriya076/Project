# Import necessary libraries
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define file paths for the saved model and TF-IDF vectorizer
model_path = 'best_model.pkl'
tfidf_vectorizer_path = 'tfidf_vectorizer.pkl'

# Load the trained model and TF-IDF vectorizer
try:
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    print(f"Model loaded from {model_path}")
    print(f"TF-IDF vectorizer loaded from {tfidf_vectorizer_path}")
except FileNotFoundError:
    print(f"Error: Ensure '{model_path}' and '{tfidf_vectorizer_path}' exist in the current directory.")
    print("Please run 03_model_training.py first to train and save the model and vectorizer.")
    exit()

# Function to clean text (should be consistent with preprocessing script)
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase and ensure it's a string
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    text = ' '.join(words)  # Join words back into a string
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to predict email subject category
def predict_email_subject(subject, body):
    # Clean the body text using the same function as in preprocessing
    cleaned_input_body = clean_text(body)
    
    # Transform the cleaned text using the loaded TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([cleaned_input_body])
    
    # Predict the subject category using the loaded model
    predicted_subject = model.predict(input_tfidf)[0]
    
    return predicted_subject

# Example usage
if __name__ == "__main__":
    print("\n--- Example Prediction ---")
    sample_subject = "Urgent: Account Locked"
    sample_body = "I cannot access my account since yesterday. Please help me unlock it as soon as possible. My username is testuser123."
    
    predicted_category = predict_email_subject(sample_subject, sample_body)
    print(f"Original Subject: {sample_subject}")
    print(f"Original Body: {sample_body}")
    print(f"Predicted Subject Category: {predicted_category}")

    print("\n--- Another Example Prediction ---")
    sample_subject_2 = "Question about billing"
    sample_body_2 = "I have a query regarding my last month's bill. It seems higher than usual. Could you please check?"

    predicted_category_2 = predict_email_subject(sample_subject_2, sample_body_2)
    print(f"Original Subject: {sample_subject_2}")
    print(f"Original Body: {sample_body_2}")
    print(f"Predicted Subject Category: {predicted_category_2}")
