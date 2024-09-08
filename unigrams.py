import os
import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')
def get_unigrams_bigrams_trigrams(text):
    tokens = nltk.word_tokenize(text.lower())
    unigrams = tokens
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    return unigrams, bigrams, trigrams


def print_sample_ngrams(unigrams, bigrams, trigrams, sample_size=5):
    print(f"Sample Unigrams: {unigrams[:sample_size]}")
    print(f"Sample Bigrams: {bigrams[:sample_size]}")
    print(f"Sample Trigrams: {trigrams[:sample_size]}")
    print()
def analyze_files_in_directory(directory_path):
    unigram_counts = []
    bigram_counts = []
    trigram_counts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                unigrams, bigrams, trigrams = get_unigrams_bigrams_trigrams(text)

                unique_unigrams = set(unigrams)
                unique_bigrams = set(bigrams)
                unique_trigrams = set(trigrams)

                unigram_counts.append(len(unique_unigrams))
                bigram_counts.append(len(unique_bigrams))
                trigram_counts.append(len(unique_trigrams))

                print(f"File: {filename}")
                print(f"Unique Unigrams: {len(unique_unigrams)}")
                print(f"Unique Bigrams: {len(unique_bigrams)}")
                print(f"Unique Trigrams: {len(unique_trigrams)}")

                # Print 5 example unigrams, bigrams, and trigrams
                print_sample_ngrams(unigrams, bigrams, trigrams)

    avg_unigrams = sum(unigram_counts) / len(unigram_counts) if unigram_counts else 0
    avg_bigrams = sum(bigram_counts) / len(bigram_counts) if bigram_counts else 0
    avg_trigrams = sum(trigram_counts) / len(trigram_counts) if trigram_counts else 0

    print(f"Average unique unigrams: {avg_unigrams}")
    print(f"Average unique bigrams: {avg_bigrams}")
    print(f"Average unique trigrams: {avg_trigrams}")

# Directory with files here
directory_path = '/Users/daniilshakirov/htrc-possiplex/htrc-possiplex/texts'
analyze_files_in_directory(directory_path)
