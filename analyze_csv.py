# Import the pandas library for data manipulation
import pandas as pd

# Define the path to the CSV file
file_path = 'ai engineer dataset.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first 5 rows of the DataFrame
print("First 5 rows of the DataFrame:")
print(df.head())
print("\n")

# Display basic information about the DataFrame, including data types and non-null values
print("Basic info about the DataFrame:")
df.info()
print("\n")

# Show the count of unique values in the 'subject' column
print("Count of unique values in the 'subject' column:")
print(df['subject'].nunique())
