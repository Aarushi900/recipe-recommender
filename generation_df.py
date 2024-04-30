import pandas as pd
import glob

# Define the directory where CSV files are stored
csv_directory = 'generation_csv/'

# Find all CSV files in the directory
csv_files = glob.glob(csv_directory + '*.csv')

# Create an empty DataFrame to hold the combined data
combined_df = pd.DataFrame()

# Iterate over all CSV files
for csv_file in csv_files:
    # Read each CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Append the data from each DataFrame to the combined DataFrame
    combined_df = combined_df._append(df, ignore_index=True)

# Specify the filename for the combined CSV file
combined_filename = 'combined_df_recipes_with_cuisine.csv'

# Save the combined DataFrame to a single CSV file
combined_df.to_csv(combined_filename, index=False)

print(f"Combined CSV file saved as {combined_filename}")
df = pd.read_csv('df_recipes_with_cuisine.csv')
df = df.dropna()
# Check for duplicates
duplicates = df[df.duplicated()]

# Optionally, remove duplicate rows
# df.drop_duplicates(inplace=True)

# Print duplicate rows if any
if not duplicates.empty:
    print("Duplicate rows found:")
    print(duplicates)
else:
    print("No duplicate rows found.")
