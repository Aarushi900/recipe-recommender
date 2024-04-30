import pandas as pd
import glob

# Specify the pattern for your saved files (replace with your actual pattern)
file_pattern = "generation_csv/df_recipes_with_cuisine_*.csv"

# Get all filenames matching the pattern
all_filenames = glob.glob(file_pattern)

# Combine all DataFrames
combined_df = pd.concat([pd.read_csv(f) for f in all_filenames])

# Save the combined DataFrame to a new file
combined_df.to_csv("final_recipes_with_cuisine.csv", index=False)
