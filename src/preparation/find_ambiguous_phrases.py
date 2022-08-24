import json
import pickle
import random
import time
from os.path import exists

import nltk.data
import pandas as pd
from scipy import sparse
from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datasets import load_dataset
from os import makedirs
from collections import Counter
from langdetect import detect


tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

import urllib3
from bs4 import BeautifulSoup

http = urllib3.PoolManager()


DATA_PATH = "../../data/"


# Step 3: Phrase with context collection
def handle_extracted_wiki_phrases_and_sentences(debug=False):
    '''
    Load a pickle file generated from extracted_wiki_phrases.py and
    merge unique phrases and combine their corresponding sentences to one list
    '''

    ambiguous_pickle_fp = CONSTRUCTION_DIR + "{}.pickle".format(FILE_NAME)

    print("Loading ambiguous examples from pickle file...")
    with open(ambiguous_pickle_fp, "rb") as pickle_file:
        examples = pickle.load(pickle_file)

    # example_dict is a dictionary with keys are phrases and values are lists of tuples (sentence, phraseType)
    example_dict = {}
    removed_phrases = []

    for example in tqdm(examples):  # example is a tuple of (phrase, sentence, phraseType, sent_idx, wiki_url)
        # Only allow ASCII characters to filter out weird/meaningless phrases
        if not example[0].isascii():
            removed_phrases.append(example[0])
            continue

        if example[0] not in example_dict:
            example_dict[example[0]] = []
        example_dict[example[0]].append((example[1], example[2], example[3], example[4]))

    print("len(examples) = {}/{}".format(len(list(set(examples))), len(examples)))
    print("len(removed_phrases) = {}/{}".format(len(list(set(removed_phrases))), len(removed_phrases)))
    print("Counter(list(set(removed_phrases))) = {}".format(Counter(list(set(removed_phrases)))))

    if debug:
        removed_phrases_subset = random.sample(removed_phrases, k=100)
        [print(phrase) for phrase in removed_phrases_subset]

        count_english_phrases = 0

        for phrase in tqdm(removed_phrases):
            try:
                count_english_phrases += 1 if detect(phrase) == "en" else 0
            except:
                print(phrase)

        print("% English phrases = {}".format(round(count_english_phrases * 1.0 / len(removed_phrases) * 100, 2)))

    # Remove duplicate sentences for each phrase and remove a phrase if it has less than 2 sentences
    example_dict = {key: list(set(values)) for key, values in example_dict.items()}
    example_dict = dict(sorted(example_dict.items(), key=lambda x: len(x[1]), reverse=True))
    new_example_dict = {key: values for (key, values) in example_dict.items() if len(values) > 1}

    print("The number of phrases BEFORE and AFTER filtering those having less than 2 sentences: {} and {}".format(len(example_dict.keys()), len(new_example_dict.keys())))
    print("Dumping all UNIQUE phrases to pickle file...")

    wiki_phrases_pickle_fp = CONSTRUCTION_DIR + "{}_filtered.pickle".format(FILE_NAME)
    with open(wiki_phrases_pickle_fp, "wb") as pickle_file:
        pickle.dump(new_example_dict, pickle_file)

    return new_example_dict


# Step 4: Find phrases of ambiguous words
def search_ambiguous_phrases_from_WiC():
    '''
    Load a dictionary of unique phrases and their lists of sentences
    to filter phrases that are not overlapped with WiC words (ambiguous)
    '''

    from datasets import load_dataset
    wic_train = load_dataset('super_glue', 'wic', split='train')
    wic_dev = load_dataset('super_glue', 'wic', split='validation')
    wic_test = load_dataset('super_glue', 'wic', split='test')

    ambiguous_word_list = [str(word) for word in list(wic_train.data['word'])]
    ambiguous_word_list.extend([str(word) for word in list(wic_dev.data['word'])])
    ambiguous_word_list.extend([str(word) for word in list(wic_test.data['word'])])
    unique_ambiguous_word_list = set(ambiguous_word_list)

    print("# Ambiguous words = {} (UNIQUE = {})".format(len(ambiguous_word_list), len(unique_ambiguous_word_list)))

    ambiguous_pickle_fp = CONSTRUCTION_DIR + "{}_filtered.pickle".format(FILE_NAME)
    print("Loading ambiguos examples from pickle file...")
    with open(ambiguous_pickle_fp, "rb") as pickle_file:
        examples = pickle.load(pickle_file)

    new_examples = {}
    for phrase, values in tqdm(examples.items()):
        if len(list(set(phrase.split(" ")).intersection(unique_ambiguous_word_list))) > 0:
            new_examples[phrase] = values

    print("Dumping all ambiguous candidates to pickle file...")
    wiki_phrases_pickle_fp = CONSTRUCTION_DIR + "{}_filtered_WiC.pickle".format(FILE_NAME)
    with open(wiki_phrases_pickle_fp, "wb") as pickle_file:
        pickle.dump(new_examples, pickle_file)

    print("# examples BEFORE and AFTER: {} / {}".format(len(examples.keys()), len(new_examples.keys())))


# Step 5a: Find phrases of distinct contexts by semantic dissimilarity
def find_ambiguous_phrases(example_dict=None, max_len_diff=1):

    # Hyperparameters
    min_threshold, max_threshold = -0.3, 0.2
    min_seq_len, max_seq_len = 5, 25
    max_sentence_capacity = 32

    ambiguous_phrase_dict = dict.fromkeys([str(x / 10.0) for x in range(-10, 11, 1)])
    for key in ambiguous_phrase_dict.keys():
        ambiguous_phrase_dict[key] = []

    # SimCSE
    model = 'princeton-nlp/sup-simcse-roberta-large'
    model = SimCSE(model)

    if not example_dict:
        print("Loading ambiguous examples from pickle file...")
        ambiguous_pickle_fp = CONSTRUCTION_DIR + "{}_filtered_WiC.pickle".format(FILE_NAME)

        # Format: example_dict = [{"phrase": ["sent1", "sent2", "sent3"]}]
        with open(ambiguous_pickle_fp, "rb") as pickle_file:
            example_dict = pickle.load(pickle_file)
            example_dict = dict(sorted(example_dict.items(), key=lambda x: len(x[1]), reverse=True))

    # For each phrase, compute cosine similarity for pairs of sentences
    # and sort them by ascending to get top_N examples starting from around -0.2
    for phrase, values in tqdm(example_dict.items()):
        sentences, phrase_types, sent_indices, wiki_urls = [], [], [], []

        for sentence, phraseType, sent_idx, wiki_url in values:
            # Since stopwords in phrases are already removed in the extract_wiki_phrases.py
            # We only limit the length of a Wiki sentence for computing similarity
            if min_seq_len <= len(sentence.split(" ")) <= max_seq_len:
                sentences.append(sentence)
                phrase_types.append(phraseType)
                sent_indices.append(sent_idx)
                wiki_urls.append(wiki_url)

        # Handle maximum 32 sentences per phrase
        n_samples = max_sentence_capacity if len(sentences) > max_sentence_capacity else len(sentences)
        selected_indices = random.sample(list(range(0, len(sentences))), k=n_samples)

        sentences = [sentences[idx] for idx in selected_indices]
        phrase_types = [phrase_types[idx] for idx in selected_indices]
        sent_indices = [sent_indices[idx] for idx in selected_indices]
        wiki_urls = [wiki_urls[idx] for idx in selected_indices]

        # Need at least 2 sentences after filtering sentence length
        if len(sentences) < 2:
            continue

        # Compute sentence embeddings
        all_sent_embs = model.encode(sentences, batch_size=512, max_length=128, return_numpy=True)

        # Compute cosine similarity for pairs of sentences
        sent_embs_sparse = sparse.csr_matrix(all_sent_embs)
        similarities_sparse = cosine_similarity(sent_embs_sparse, dense_output=False)
        similarities_sparse = similarities_sparse.tocoo()

        for sent_idx1, sent_idx2, score in zip(similarities_sparse.row, similarities_sparse.col, similarities_sparse.data):
            # Avoid adding duplicate pairs in the sparse matrix
            if sent_idx2 <= sent_idx1:
                break

            # Analyze examples across different cosine similarity scores ranging from -1 to 1 with
            # (1) the cosine similarity score within the allowed range [min_threshold, max_threshold] and
            # (2) the difference in length between 2 sentences does not exceed `max_len_diff`
            if min_threshold <= score <= max_threshold:
                len_diff = abs(len(sentences[sent_idx1].split(" ")) - len(sentences[sent_idx2].split(" ")))

                if len_diff <= max_len_diff:
                    key = "0.0" if str(round(score, 1)) == "-0.0" else str(round(score, 1))
                    ambiguous_phrase_dict[key].append((phrase, sentences[sent_idx1], sentences[sent_idx2],
                                                       score, phrase_types[sent_idx1], phrase_types[sent_idx2],
                                                       sent_indices[sent_idx1], sent_indices[sent_idx2],
                                                       wiki_urls[sent_idx1], wiki_urls[sent_idx2],))

    print("Dumping all ambiguous candidates to pickle file...")
    out_file_name = "ambiguous_candidates_{}_{}_{}_{}_{}_{}".format(min_threshold, max_threshold, min_seq_len, max_seq_len, max_sentence_capacity, max_len_diff)
    wiki_phrases_pickle_fp = CONSTRUCTION_DIR + "{}.pickle".format(out_file_name)
    with open(wiki_phrases_pickle_fp, "wb") as pickle_file:
        pickle.dump(ambiguous_phrase_dict, pickle_file)

    print("Finished!!!")


# Step 5b: Find phrases of distinct contexts by domain dissimilarity
def read_output_files_and_sort_for_AMT():
    def prepare_wiki_articles():
        # Prepare Wiki sentences by checking if the preprocessed wiki file exists to load
        # Otherwise, load the original wiki file
        start_time = time.time()
        wiki_fp = DATA_PATH + "wiki/wikiextractor/enwiki-latest-pages-articles.txt"
        wiki_pickle_fp = DATA_PATH + "wiki/wikiextractor/wiki_objects_filtered.pickle"

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

        # Create a list of wiki articles for speeding up the searching process
        wiki_article_dict = {wiki_obj["url"]: wiki_obj["text"] for wiki_obj in tqdm(wiki_objs)}
        return wiki_article_dict

    def find_categories(wiki_url):
        '''
        ThangPM: Extract categories directly from a Wiki page
        '''

        response = http.request('GET', wiki_url)
        soup = BeautifulSoup(response.data.decode('utf-8'), features="lxml")

        # Get only visible categories and ignore the hidden ones
        cat_div = soup.find("div", {"id": "mw-normal-catlinks"})
        if not cat_div:
            return []

        categories = [a.attrs['href'].split("Category:")[-1] for a in cat_div.find_all('a', href=True) if
                      a.attrs['href'].startswith("/wiki/Category")]

        return categories

    wic_train = load_dataset('super_glue', 'wic', split='train')
    wic_dev = load_dataset('super_glue', 'wic', split='validation')
    wic_test = load_dataset('super_glue', 'wic', split='test')

    ambiguous_word_list = [str(word) for word in list(wic_train.data['word'])]
    ambiguous_word_list.extend([str(word) for word in list(wic_dev.data['word'])])
    ambiguous_word_list.extend([str(word) for word in list(wic_test.data['word'])])
    unique_ambiguous_word_list = set(ambiguous_word_list)

    # SimCSE
    model = SimCSE('princeton-nlp/sup-simcse-roberta-large')

    wiki_article_dict = prepare_wiki_articles()
    wiki_cache = {}
    number_extra_sentences = 2

    amt_data = {"phrase": [],
                "original_sentence1": [], "original_sentence2": [],
                "sentence1": [], "sentence2": [],
                "pre_context1": [], "pre_context2": [],
                "post_context1": [], "post_context2": [],
                "sent_idx1": [], "sent_idx2": [],
                "wiki_url1": [], "wiki_url2": [],
                "category1": [], "category2": [],
                "semantic_score": [], "domain_score": [],
                "phrase_type": [], "wic_word": []}

    skip_count = 0
    count_double_propn = 0
    phrase_list = []

    with open(CONSTRUCTION_DIR + "ambiguous_candidates_-0.3_0.2_5_25_32_3.pickle", "rb") as pickle_file:
        ambiguous_phrase_dict = pickle.load(pickle_file)

    for values_per_key in ambiguous_phrase_dict.values():
        if len(values_per_key) > 0:
            for values in values_per_key:
                phrase, sentence1, sentence2, score, phrase_type1, phrase_type2, sent_idx1, sent_idx2, wiki_url1, wiki_url2 = values
                phrase_list.append((phrase, sentence1, sentence2, sent_idx1, sent_idx2, wiki_url1, wiki_url2, score, [phrase_type1, phrase_type2]))

    # Sorted the list by cosine similarity scores
    phrase_list = sorted(phrase_list, key=lambda x: x[-2], reverse=False)

    # ThangPM: Load a dictionary of Wiki categories given Wiki articles (key=url, value=categories) which was
    # created in wiki_category_crawling.py for computing cosine similarity scores between domain embeddings.
    wiki_categories_fp = CONSTRUCTION_DIR + "unique_wiki_urls_with_categories.pickle"
    if not exists(wiki_categories_fp):
        all_wiki_url_list = []
        for candidate in tqdm(phrase_list):
            wiki_url1 = candidate[5]
            wiki_url2 = candidate[6]
            all_wiki_url_list.extend([wiki_url1, wiki_url2])

        # ThangPM: Store a list of unique wiki URLs for crawling categories via API
        print("Dumping unique Wiki URL list to pickle file...")
        with open(CONSTRUCTION_DIR + "unique_wiki_url_list.pickle", "wb") as pickle_file:
            pickle.dump(list(set(all_wiki_url_list)), pickle_file)

        print("Please run 'wiki_category_crawling.py' first to save the categories for the selected Wiki articles and re-run this function!")
        exit(0)
    else:
        with open(wiki_categories_fp, "rb") as pickle_file:
            wiki_category_dict = pickle.load(pickle_file)

    all_categories1, all_categories2 = [], []
    categories_cache = {}

    # ThangPM: In case the preloaded cache misses, load categories from this cache.
    if exists(CONSTRUCTION_DIR + "wiki_categories_dict_cache.pickle"):
        with open(CONSTRUCTION_DIR + "wiki_categories_dict_cache.pickle", "rb") as pickle_file:
            categories_cache = pickle.load(pickle_file)

    print("Handling list of Wiki categories...")
    for candidate in tqdm(phrase_list):
        wiki_url1 = candidate[5]
        wiki_url2 = candidate[6]

        categories1 = categories_cache[wiki_url1] if wiki_url1 in categories_cache else (wiki_category_dict[wiki_url1] if wiki_url1 in wiki_category_dict else [])
        categories2 = categories_cache[wiki_url2] if wiki_url2 in categories_cache else (wiki_category_dict[wiki_url2] if wiki_url2 in wiki_category_dict else [])

        # ThangPM: Call Wiki API to get categories for Wiki articles if not existed in the preloaded cache.
        if len(categories1) == 0:
            categories1 = find_categories(wiki_url1)
            categories_cache[wiki_url1] = categories1
        if len(categories2) == 0:
            categories2 = find_categories(wiki_url2)
            categories_cache[wiki_url2] = categories2

        all_categories1.append(", ".join([category.replace("_", " ") for category in categories1]))
        all_categories2.append(", ".join([category.replace("_", " ") for category in categories2]))

    print("Dumping Wiki categories to pickle file...")
    with open(CONSTRUCTION_DIR + "wiki_categories_dict_cache.pickle", "wb") as pickle_file:
        pickle.dump(categories_cache, pickle_file)

    print("Computing cosine distances of domain embeddings...")

    batch_size = 2048
    n_batches = int(len(phrase_list) / batch_size) + 1
    all_domain_scores = []

    for i in tqdm(range(n_batches)):
        categories1 = all_categories1[i*batch_size: (i+1)*batch_size]
        categories2 = all_categories2[i*batch_size: (i+1)*batch_size]
        category_embs1 = model.encode(categories1, batch_size=batch_size, max_length=128, return_numpy=True)
        category_embs2 = model.encode(categories2, batch_size=batch_size, max_length=128, return_numpy=True)

        domain_scores = cosine_similarity(category_embs1, category_embs2)
        all_domain_scores.extend([domain_score[idx] for idx, domain_score in enumerate(domain_scores)])

    for candidate, domain_score in tqdm(zip(phrase_list, all_domain_scores)):
        phrase = candidate[0]
        sentence1 = candidate[1]
        sentence2 = candidate[2]
        sent_idx1 = candidate[3]
        sent_idx2 = candidate[4]
        wiki_url1 = candidate[5]
        wiki_url2 = candidate[6]
        semantic_score = candidate[7]
        phrase_types = candidate[8]

        if wiki_url1 in wiki_cache:
            sentences1 = wiki_cache[wiki_url1]
        else:
            sentences1 = tokenizer.tokenize(wiki_article_dict[wiki_url1])
            wiki_cache[wiki_url1] = sentences1

        if wiki_url2 in wiki_cache:
            sentences2 = wiki_cache[wiki_url2]
        else:
            sentences2 = tokenizer.tokenize(wiki_article_dict[wiki_url2])
            wiki_cache[wiki_url2] = sentences2

        pre_context1 = sentences1[max(0, sent_idx1-number_extra_sentences): sent_idx1]
        post_context1 = sentences1[sent_idx1+1: sent_idx1+number_extra_sentences]
        pre_context2 = sentences2[max(0, sent_idx2-number_extra_sentences): sent_idx2]
        post_context2 = sentences2[sent_idx2+1: sent_idx2+number_extra_sentences]

        # ThangPM: Find indices of the phrases in their associated sentences
        # Need to call lower() because SpaCy returns all noun phrases in lowercase.
        phrase_idx1 = sentence1.lower().find(phrase)
        phrase_idx2 = sentence2.lower().find(phrase)

        # Ignore those examples whose phrase cannot be found due to "\n" character
        if phrase_idx1 == -1 or phrase_idx2 == -1:
            skip_count += 1
            continue

        phrase_highlight = '<span style="background-color:yellow;">{}</span>'.format(phrase)
        sentence1_highlight = sentence1[:phrase_idx1] + phrase_highlight + sentence1[phrase_idx1 + len(phrase):]
        sentence2_highlight = sentence2[:phrase_idx2] + phrase_highlight + sentence2[phrase_idx2 + len(phrase):]

        categories1 = categories_cache[wiki_url1] if wiki_url1 in categories_cache else (wiki_category_dict[wiki_url1] if wiki_url1 in wiki_category_dict else [])
        categories2 = categories_cache[wiki_url2] if wiki_url2 in categories_cache else (wiki_category_dict[wiki_url2] if wiki_url2 in wiki_category_dict else [])

        amt_data["phrase"].append(phrase)
        amt_data["sentence1"].append(sentence1_highlight)
        amt_data["sentence2"].append(sentence2_highlight)
        amt_data["original_sentence1"].append(sentence1)
        amt_data["original_sentence2"].append(sentence2)
        amt_data["pre_context1"].append(" ".join(pre_context1))
        amt_data["pre_context2"].append(" ".join(pre_context2))
        amt_data["post_context1"].append(" ".join(post_context1))
        amt_data["post_context2"].append(" ".join(post_context2))
        amt_data["semantic_score"].append(semantic_score)
        amt_data["domain_score"].append(domain_score)
        amt_data["sent_idx1"].append(sent_idx1)
        amt_data["sent_idx2"].append(sent_idx2)
        amt_data["wiki_url1"].append(wiki_url1)
        amt_data["wiki_url2"].append(wiki_url2)
        amt_data["category1"].append(categories1)
        amt_data["category2"].append(categories2)
        amt_data["phrase_type"].append(phrase_types)
        amt_data["wic_word"].append(list(set(phrase.split(" ")).intersection(unique_ambiguous_word_list)))

    pd.DataFrame.from_dict(amt_data).sort_values(by=["domain_score"], ascending=True).to_csv(CONSTRUCTION_DIR + "amt_data_sorted_by_domain_new_full_latest.csv", index=False)
    print("Finished!!! Skip {} examples due to '\\n' issue and {} examples for ['PROPN', 'PROPN'].".format(skip_count, count_double_propn))


# Step 6: Remove proper nouns
def generate_ultimate_phrase_list(size=19500):

    df = pd.read_csv(CONSTRUCTION_DIR + "amt_data_sorted_by_domain_new_full_latest.csv")
    print("The number of rows = {}".format(len(df)))

    # This case is less likely to be ambiguous so they should be excluded
    df = df.loc[~(df["phrase_type"] == "['PROPN', 'PROPN']")]

    # Clean malformed phrases starting with a double quote: 2426 examples
    df_malformed_phrase = df["phrase"].str.extract(r'(^".*)').dropna(how='all')
    for idx, row in tqdm(df_malformed_phrase.iterrows()):
        assert df["phrase"].loc[idx] == row[0]
        if row[0].count('"') == 1 or row[0].startswith('""'):
            df.loc[idx, "phrase"] = row[0].replace("\"", "").strip()

    # Remove malformed sentences
    df = df[df.apply(lambda x: (~x.astype(str).str.contains(".* \\.", case=True, regex=True, na=False)))].dropna()

    # P3. Replace control characters by an empty string
    del_pattern = r'[\x80-\x99]'

    df['sentence1'] = df['sentence1'].str.replace(del_pattern, '', regex=True).astype(str)
    df['sentence2'] = df['sentence2'].str.replace(del_pattern, '', regex=True).astype(str)
    df['pre_context1'] = df['pre_context1'].str.replace(del_pattern, '', regex=True).astype(str)
    df['pre_context2'] = df['pre_context2'].str.replace(del_pattern, '', regex=True).astype(str)
    df['post_context1'] = df['post_context1'].str.replace(del_pattern, '', regex=True).astype(str)
    df['post_context2'] = df['post_context2'].str.replace(del_pattern, '', regex=True).astype(str)

    df.to_csv(CONSTRUCTION_DIR + "amt_data_sorted_by_domain_new_full_latest_final.csv")
    df.head(size).to_csv(CONSTRUCTION_DIR + "amt_data_{}_sorted_by_semantic_domain_latest.csv".format(size), index=False)


if __name__ == '__main__':

    FILE_NAME = "unique_wiki_phrases_wo_categories"
    CONSTRUCTION_DIR = DATA_PATH + "preparation/"

    # ------------------------------------------------------------
    # Workflow to generate ambiguous examples for annotation
    # ------------------------------------------------------------
    if not exists(CONSTRUCTION_DIR):
        makedirs(CONSTRUCTION_DIR)

    handle_extracted_wiki_phrases_and_sentences()
    search_ambiguous_phrases_from_WiC()
    find_ambiguous_phrases(max_len_diff=3)
    read_output_files_and_sort_for_AMT()

    generate_ultimate_phrase_list(size=19500)
    # ------------------------------------------------------------



