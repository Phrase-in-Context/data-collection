import itertools
import json
import pickle
import time
import re
from multiprocessing import Pool
from os.path import exists
from tqdm import tqdm
from collections import Counter

import nltk
nltk.download('punkt')
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
from nltk.corpus import stopwords

import spacy
nlp_spacy = spacy.load("en_core_web_sm")
nlp_spacy.add_pipe("merge_noun_chunks")
nlp_spacy.max_length = 2000000

n_gram_range = (2, 5)
stop_words_en = "english"

caches = {}                 # Caches to store categories for Wiki URLs
number_extra_sentences = 2  # Number of sentences added before and after each context sentence
cached_stop_words = Counter(stopwords.words('english'))


# Step 2: Phrase extraction
def process_wiki_articles(pid, articles):
    all_candidates = []
    tqdm_text = "#" + "{}".format(pid).zfill(3)

    for article in tqdm(articles, total=len(articles), desc=tqdm_text, position=pid):
        sentences = tokenizer.tokenize(article["text"])

        for idx, sentence in enumerate(sentences):
            doc = nlp_spacy(sentence)
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
            prop_nouns = [token.text.lower() for token in doc if token.pos_ == "PROPN"]
            nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
            ambiguous_candidates = []

            for phrase in list(set(noun_phrases + prop_nouns + nouns)):
                phrase_type = "PROPN" if phrase in prop_nouns else "NOUN"

                processed_phrase = re.sub(r"^\W+", "", phrase.strip().replace("\n", " "))
                phrase_tokens = processed_phrase.strip().split(" ")
                if len(phrase_tokens) < 2:
                    continue

                # a. Remove stop words at the beginning and at the end
                # phrase_tokens = [token for token in phrase_tokens if token not in cached_stop_words]
                start_idx, end_idx = 0, len(phrase_tokens) - 1
                while start_idx < len(phrase_tokens) and phrase_tokens[start_idx] in cached_stop_words:
                    start_idx += 1
                while end_idx >= 0 and phrase_tokens[end_idx] in cached_stop_words:
                    end_idx -= 1

                phrase_tokens = phrase_tokens[start_idx: end_idx + 1] if start_idx < end_idx + 1 else []

                # b. Only accept pairs whose phrases are at least 2-token long.
                if len(phrase_tokens) >= 2:
                    candidate = (" ".join(phrase_tokens), sentence, phrase_type, idx, article["url"])
                    ambiguous_candidates.append(candidate)

            all_candidates.extend(ambiguous_candidates)

    return all_candidates


if __name__ == '__main__':
    # Prepare Wiki sentences by checking if the preprocessed wiki file exists to load
    # Otherwise, load the original wiki file
    start_time = time.time()
    wiki_fp = "../data/wiki/wikiextractor/enwiki-latest-pages-articles.txt"
    wiki_pickle_fp = "../data/wiki/wikiextractor/wiki_objects_filtered.pickle"

    # Step 1: Data preparation
    if exists(wiki_pickle_fp):
        with open(wiki_pickle_fp, "rb") as pickle_file:
            print("Loading Wiki objects from pickle file...")
            wiki_objs = pickle.load(pickle_file)
            execution_time = time.time() - start_time
            print("Took {:.2f}s to load the wiki file.".format(execution_time))
    else:
        print("Loading Wiki objects from text file...")
        wiki_file = open(wiki_fp, "r")
        wiki_objs = [json.loads(line) for line in wiki_file]
        wiki_file.close()
        execution_time = time.time() - start_time
        print("Took {:.2f}s to load the wiki file.".format(execution_time))
        print("All Wiki objects = {}".format(len(wiki_objs)))

        # Filter empty-text wiki objects and create a new wiki file for faster process
        wiki_objs = [wiki_obj for wiki_obj in wiki_objs if wiki_obj["text"]]
        print("Filtered Wiki objects = {}".format(len(wiki_objs)))

        # Dump filtered wiki objects to the wiki pickle file
        with open(wiki_pickle_fp, "wb") as pickle_file:
            print("Dumping Wiki objects to pickle file...")
            pickle.dump(wiki_objs, pickle_file)

        print("Finish loading Wiki objects from pickle file...")

    # Start extraction
    print("Processing wiki articles...")
    cpu_cores = 90
    batch_size = int(len(wiki_objs) / (cpu_cores - 1))
    sub_lists = []

    for i in range(cpu_cores):
        sub_wiki_articles = wiki_objs[i * batch_size: (i + 1) * batch_size]
        sub_lists.append((i, sub_wiki_articles))

    pool = Pool(processes=cpu_cores)
    all_phrases = pool.starmap(process_wiki_articles, sub_lists)  # Use starmap for multiple arguments
    all_phrases = list(itertools.chain(*all_phrases))

    print("Number of phrases were generated: {}".format(len(all_phrases)))
    print("Number of UNIQUE phrases were generated: {}".format(len(set(all_phrases))))

    print("Dumping all UNIQUE phrases to pickle file...")
    wiki_phrases_pickle_fp = "../data/construction/unique_wiki_phrases_wo_categories.pickle"
    with open(wiki_phrases_pickle_fp, "wb") as pickle_file:
        pickle.dump(all_phrases, pickle_file)
