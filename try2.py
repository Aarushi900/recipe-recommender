
import unidecode
import ast
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
# from Detector import detect_objects
from ing_parser import ingredient_parser, ingredient_parser1

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed.values:
        doc = sorted(doc)
        # print(doc)
        corpus_sorted.append(doc)
    return corpus_sorted


def title_parser(title):
    title = unidecode.unidecode(title)
    return title


def ingredient_parser_final(ingredient):
    """
    neaten the ingredients being outputted
    """
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)

    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):

        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, docs):  # comply with scikit-learn transformer requirement
        """
		Fit in a list of docs, which had been preprocessed and tokenized,
		such as word bi-grammed, stop-words removed, lemmatized, part of speech filtered.
		Then build up a tfidf model to compute each word's idf as its weight.
		Noted that tf weight is already involved when constructing average word vectors, and thus omitted.
		:param
			pre_processed_docs: list of docs, which are tokenized
		:return:
			self
		"""

        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)  # must be list of text string

        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)  # used as default value for defaultdict
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent):
        """
		Compute average word vector for a single doc/sentence.
		:param sent: list of sentence tokens
		:return:
			mean: float of averaging word vectors
		"""

        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(
                    self.word_model.wv.get_vector(word) * self.word_idf_weight[word]
                )  # idf weighted

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            # logging.warning(
            #     "cannot compute average owing to no vector for {}".format(sent)
            # )
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def word_average_list(self, docs):
        """
		Compute average word vector for multiple docs, where docs had been tokenized.
		:param docs: list of sentence in list of separated tokens
		:return:
			array of average word vector in shape (len(docs),)
		"""
        return np.vstack([self.word_average(sent) for sent in docs])
    

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

def get_recommendations(N, scores, cuisine,df_recipes):
    # order the scores with and filter to get the highest N scores
    top = sorted(range(min(len(scores), df_recipes.shape[0])), key=lambda i: scores[i], reverse=True)[:N]
    return df_recipes.iloc[top]


def get_recs(ingredients, N,cuisine, mean = False):
    # load in word2vec model
    model = Word2Vec.load("store/model_cbow1.bin")
    model.init_sims(replace=True)
    if model:
        print("Successfully loaded model")
    # load in data
    data = pd.read_csv("store/df_recipes_parsed_final.csv")
    
    data = data[data["cuisine"] == cuisine]
    # parse ingredients
    data["parsed"] = data.ingredients.apply(ingredient_parser1)
    # create corpus
    corpus = get_and_sort_corpus(data)
    

    # use TF-IDF as weights for each word embedding
    tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
    tfidf_vec_tr.fit(corpus)
    doc_vec = tfidf_vec_tr.transform(corpus)
    doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
    assert len(doc_vec) == len(corpus)

    # create embessing for input text
    input = ingredients
    # create tokens with elements
    input = input.split(",")
    # parse ingredient list
    input = ingredient_parser(input)
    # get embeddings for ingredient doc
    
    input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    # get cosine similarity between input embedding and all the document embeddings
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    # Filter top N recommendations
    recommendations = get_recommendations(N, scores, cuisine, data)
    return recommendations


if __name__ == "__main__":
    # image_path = "cap.jpeg"
    # detected_labels = detect_objects(image_path)

    cuisines = ["Mexican", "Indian", "Italian", "American" ,"Japanese", "Thai", "French", "Chinese", "Unknown"]

    print(cuisines)
    test_ingredients = "capsicum"
    cuisine = input("Enter the cuisine: ")
    N= int(input("Enter the number of recommendations: "))
    # recs = get_recs(detected_labels, cuisine, N)
    recs = get_recs(test_ingredients, N,cuisine)
    print(f'Top 5 Recommendations for {cuisine} Cuisine: ')
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
    #   missing_ingredients_list = missing_ingredients(detected_labels, ingredients)
      missing_ingredients_list = missing_ingredients(test_ingredients, ingredients)
      print('Missing Ingredients: ', missing_ingredients_list)
      print()