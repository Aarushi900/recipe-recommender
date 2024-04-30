import google.generativeai as genai
import os
import pandas as pd
from config import API_KEY

genai.configure(api_key=API_KEY)
df = pd.read_csv('store/df_recipes.csv')
safety_settings = [
  {
    "category": "HARM_CATEGORY_DANGEROUS",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
     "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]

model = genai.GenerativeModel('models/gemini-pro',safety_settings=safety_settings)

def predict_cuisine(recipe_name):
  """Predicts the cuisine of a single recipe name using Generative AI.

  Args:
    recipe_name (str): The name of the recipe for which to predict cuisine.

  Returns:
    str: The predicted cuisine.
  """

  prompt = f"type of cuisine of {recipe_name} from given list( Mexican, Indian, Italian, American ,Japanese, Thai, French, Chinese, Unknown )...one word answer only"
  response = model.generate_content(prompt)
  return response.text.strip() # Remove leading/trailing whitespace

# Specify the starting index for predictions
start_index = 0 # Replace with the desired starting index (e.g., 3 for starting from the 4th row)

# Optional: Specify a filename for the intermediate DataFrame
output_filename = 'store/df_recipes_with_cuisine.csv'  # Replace with desired filename

predicted_cuisines = []
batch_size = 100
for start_index in range(0, len(df), batch_size):
    end_index = min(start_index + batch_size, len(df))
    batch_df = df.iloc[start_index:end_index]

    # Predict cuisines for the batch
    predicted_cuisines = []
    for recipe_name in batch_df['recipe_name']:
        predicted_cuisine = predict_cuisine(recipe_name)
        predicted_cuisines.append(predicted_cuisine)

    # Update the batch dataframe using .loc
    batch_df.loc[:, 'cuisine'] = predicted_cuisines

    # Save the batch dataframe (optional with incrementing filename)
    if start_index % 100 == 0:
        filename = f"{output_filename}_{start_index}.csv"
        batch_df.to_csv(filename, index=False)
        print(f"Saved intermediate results to {filename} (rows {start_index} to {end_index-1})")
