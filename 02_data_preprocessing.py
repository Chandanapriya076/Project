# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load the dataset
file_path = 'ai engineer dataset.csv'
df = pd.read_csv(file_path)

print("Original DataFrame head:")
print(df.head())
print("\n")

# 1. Check for missing values
print("Missing values before preprocessing:")
print(df.isnull().sum())
print("\n")

# Fill missing values in 'subject' and 'body' with empty strings if any exist
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')

# 2. Clean the text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply the cleaning function to the 'body' column to create 'cleaned_body'
df['cleaned_body'] = df['body'].apply(clean_text)

print("DataFrame head after text cleaning:")
print(df.head())
print("\n")

# 3. Analyze class imbalance with a bar chart
plt.figure(figsize=(12, 6))
df['subject'].value_counts().plot(kind='bar')
plt.title('Distribution of Subject Categories')
plt.xlabel('Subject Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('subject_distribution.png') # Save the plot
plt.show() # Display the plot

print("Class distribution plot saved as 'subject_distribution.png' and displayed.")
print("\n")

# 4. Save the cleaned data
output_file_path = 'cleaned_dataset.csv'
df.to_csv(output_file_path, index=False)
print(f"Cleaned data saved to {output_file_path}")
