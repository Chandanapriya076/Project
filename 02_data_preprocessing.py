# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re # For removing extra spaces

# 2. Download NLTK data: 'punkt', 'stopwords'
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Note: 'punkt_tab' is not a standard NLTK corpus. Assuming it was a typo for 'punkt' or a custom corpus.
# If it's a custom corpus, it needs to be provided separately.

print("NLTK 'punkt' and 'stopwords' resources checked/downloaded.")
print("\n")

# 3. Load the CSV and show the first 5 rows
file_path = 'ai engineer dataset.csv'
df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("First 5 rows of the original DataFrame:")
print(df.head())
print("\n")

# 4. Check for missing values
print("Missing values before preprocessing:")
print(df.isnull().sum())
print("\n")

# Fill missing values in 'body' with empty strings to prevent errors during text cleaning
df['body'] = df['body'].fillna('')
print("Missing values in 'body' column filled with empty strings.")
print("\n")

# 5. Create a function to clean the 'body' text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    text = ' '.join(words) # Join words back into a string
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

print("Text cleaning function defined.")
print("\n")

# 6. Apply the function to create a new 'cleaned_body' column
df['cleaned_body'] = df['body'].apply(clean_text)
print("'cleaned_body' column created by applying text cleaning function.")
print("\n")

# 7. Show the first 5 rows after cleaning
print("First 5 rows of the DataFrame after text cleaning:")
print(df[['body', 'cleaned_body']].head())
print("\n")

# 8. Create a bar chart of value counts for the 'subject' column
plt.figure(figsize=(12, 7))
df['subject'].value_counts().plot(kind='bar')
plt.title('Distribution of Email Subjects')
plt.xlabel('Subject Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('subject_distribution.png')
print("Bar chart 'subject_distribution.png' created and saved.")
plt.show()
print("Bar chart displayed.")
print("\n")

# 9. Save the cleaned DataFrame to 'cleaned_dataset.csv' without the index
output_file_path = 'cleaned_dataset.csv'
df.to_csv(output_file_path, index=False)
print(f"Cleaned DataFrame saved to {output_file_path}")
