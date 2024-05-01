

import numpy as np
import pandas as pd 
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

import config
from ing_parser import ingredient_parser1

data = pd.read_csv('store/df_recipes_parsed_final.csv')
data['parsed_new'] = data.ingredients.apply(ingredient_parser1)
print(data.head())
# get corpus with the documents sorted in alphabetical order
def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed_new.values:
        doc = sorted(doc)
        # print(doc)
        corpus_sorted.append(doc)
    return corpus_sorted

corpus = get_and_sort_corpus(data)
# print(corpus[:5])
print(f"Length of corpus: {len(corpus)}")
# calculate average length of each document 
lengths = [len(doc) for doc in corpus]
avg_len = float(sum(lengths)) / len(lengths)
avg_len
# train word2vec model 
sg = 0 # CBOW: build a language model tha   t correctly predicts the center word given the context words in which the center word appears
workers = 2 # number of CPUs
window = 6 # window size: average length of each document 
min_count = 1 # unique ingredients are important to decide recipes 
# Create the Word2Vec model using CBOW (continuous bag-of-words)

# Train the model

model_cbow = Word2Vec(corpus, sg=sg, workers=workers, window=window, min_count=min_count, vector_size=50)
model_cbow.train(corpus, total_examples=model_cbow.corpus_count, epochs=10)
#Summarize the loaded model
print(model_cbow)

#Summarize vocabulary
words = list(model_cbow.wv.index_to_key)
words.sort()
print(words)
# save model
model_cbow.save('store/model_cbow1.bin')
	
loaded_model = Word2Vec.load('store/model_cbow1.bin')
if loaded_model:
    print("Successfully loaded model")

similar_words = loaded_model.wv.most_similar(positive=['chicken'], topn=10)
print(similar_words)
