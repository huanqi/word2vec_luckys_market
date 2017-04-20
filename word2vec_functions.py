# aggregate reviews for specific key words
import pandas as pd

def key_word_lookup(reviews, model, key_word):
    # gather the similar words with value > 0.5
    similar_words = model.most_similar(key_word, topn = 10)
    similar_words = [obj[0] for obj in similar_words if obj[1] > 0.5]
    
    key_word_list = [None] * len(reviews)
    similar_word_list = [None] * len(reviews)
    for i, review in enumerate(reviews):
        if key_word in review:
            key_word_list[i] = key_word
        
        temp_similar_word_list = [word for word in similar_words if word in review]
        if len(temp_similar_word_list) > 0:
            similar_word_list[i] = temp_similar_word_list
    
    result = pd.DataFrame({'key_word': key_word_list, 'similar_word': similar_word_list, \
                          'reviews': reviews}, 
                         columns = ['key_word', 'similar_word', 'reviews'])
            
    result = result[~result['key_word'].isnull() | ~result['similar_word'].isnull()]
    return result

# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, 'lxml').get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Download the punkt tokenizer for sentence splitting
import nltk.data
# nltk.download()   
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences