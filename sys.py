import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from ing_parser import ingredient_parser
import pickle
import config 
import unidecode, ast

def get_recommendations(N, scores, cuisine,df_recipes):
    # order the scores with and filter to get the highest N scores
    top = sorted(range(min(len(scores), df_recipes.shape[0])), key=lambda i: scores[i], reverse=True)[:N]
    return df_recipes.iloc[top]

def RecSys(ingredients, cuisine, N=5):
    """
    The recommendation system takes in a list of ingredients and a cuisine, 
    and returns a list of top 5 recipes of that cuisine based on cosine similarity. 
    :param ingredients: a list of ingredients
    :param cuisine: the desired cuisine
    :param N: the number of recommendations returned 
    :return: top 5 recommendations for cooking recipes of the specified cuisine
    """

    # Load in tfidf model and encodings 
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = pickle.load(f)

    with open(config.TFIDF_MODEL_PATH, "rb") as f:
        tfidf = pickle.load(f)

    # Parse the ingredients using my ingredient_parser 
    try: 
        ingredients_parsed = ingredient_parser(ingredients)
    except:
        ingredients_parsed = ingredient_parser([ingredients])
    
    # Use the pretrained tfidf model to encode the input ingredients
    ingredients_tfidf = tfidf.transform([ingredients_parsed])

    # Calculate cosine similarity between actual recipe ingredients and test ingredients
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)

    # Filter top N recommendations for the specified cuisine
    df_recipes = pd.read_csv('store\df_recipes_parsed1.csv')

    
    df_recipes = df_recipes[df_recipes["cuisine"] == cuisine]

    recommendations = get_recommendations(N, scores, cuisine,df_recipes)
    return recommendations


def missing_ingredients(input_ingredients, recommended_ingredients):
    """
    Find the ingredients from recommended recipes that are missing in the input list.
    :param input_ingredients: a string or list of input ingredients
    :param recommended_ingredients: a string containing comma-separated ingredients of a recommended recipe
    :return: a list of missing ingredients
    """
    # Convert input ingredients to a set for faster lookup
    if isinstance(input_ingredients, str):
        input_ingredients = set(input_ingredients.split(','))
    else:
        input_ingredients = set(input_ingredients)

    # Split recommended ingredients and convert to a set
    recommended_ingredients = set(recommended_ingredients.split(','))

    # Find the missing ingredients
    missing = recommended_ingredients - input_ingredients
    return list(missing)


if __name__ == "__main__":
    # test ingredients
    test_ingredients = "pasta, tomato, onion"
    cuisine = input("Enter the cuisine: ")
    N= int(input("Enter the number of recommendations: "))
    recs = RecSys(test_ingredients, cuisine, N)
    print('Top 5 Recommendations for Italian Cuisine: ')
    print(len(recs))
    print()
    for index, row in recs.iterrows():
      recipe_name = row['recipe_name']
      print(recipe_name)  # Replace with actual column name
      url = row['recipe_urls']
      print('Recipe URL: ')   
      print(url)
      ingredients = row['ingredients']
      print('Ingredients: ')
      print(ingredients)
      missing_ingredients_list = missing_ingredients(test_ingredients, ingredients)
      print('Missing Ingredients: ', missing_ingredients_list)
      print()