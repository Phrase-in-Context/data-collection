import numpy as np
import pandas as pd
import re
import random
import json
import pickle
import time
import math
from os.path import exists
from tqdm import tqdm
import itertools
from collections import Counter

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

import requests
word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
WORDS = requests.get(word_site).content.splitlines()

AMT_DATA_PATH = "../data/amt/"
UPWORK_DATA_PATH = "../data/upwork/"
UPWORK_RESULTS_PATH = "../results/upwork/"
DATASETS_PATH = "../datasets/"


def prepare_wiki_articles():
    # Prepare Wiki sentences by checking if the preprocessed wiki file exists to load
    # Otherwise, load the original wiki file
    start_time = time.time()
    wiki_fp = "../data/wiki/wikiextractor/enwiki-latest-pages-articles.txt"
    wiki_pickle_fp = "../data/wiki/wikiextractor/wiki_objects_filtered.pickle"

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
    # wiki_article_dict = {wiki_obj["url"]: tokenizer.tokenize(wiki_obj["text"]) for wiki_obj in tqdm(wiki_objs)}
    wiki_article_dict = {wiki_obj["url"]: wiki_obj["text"] for wiki_obj in tqdm(wiki_objs)}
    return wiki_article_dict


def generate_puss_datasets():

    def generate_csv_files(list_examples, train_ids, dev_ids, test_ids, setting):
        train_dataset = [example for example in list_examples if example[0] in train_ids]
        dev_dataset = [example for example in list_examples if example[0] in dev_ids]
        test_dataset = [example for example in list_examples if example[0] in test_ids]

        random.shuffle(train_dataset)
        random.shuffle(dev_dataset)
        random.shuffle(test_dataset)

        print("len(train) / len(dev) / len(test) = {} / {} / {}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))

        df_train = pd.DataFrame(train_dataset, columns=["ID", "context", "query", "answer", "answer_start", "url"])
        df_dev = pd.DataFrame(dev_dataset, columns=["ID", "context", "query", "answer", "answer_start", "url"])
        df_test = pd.DataFrame(test_dataset, columns=["ID", "context", "query", "answer", "answer_start", "url"])

        df_train.to_csv(version + "/" + "train.csv", header=False, index=False)
        df_dev.to_csv(version + "/" + "dev.csv", header=False, index=False)
        df_test.to_csv(version + "/" + "test.csv", header=False, index=False)

    df = pd.read_csv(DATASETS_PATH + "data_15K_pairs_fully_verified_official.csv", lineterminator='\n')
    wiki_article_dict = prepare_wiki_articles()

    random.seed(42)

    list_shared_ids = []
    list_full_examples = []
    list_short_examples = []
    short_context_sentences = 11

    for idx, row in tqdm(df.iterrows()):
        for local_idx in [1, 2]:
            sent_idx = row["sent_idx{}".format(local_idx)]

            selected = row["Q{}_AMT_select".format(local_idx)] == "Yes" or row["Q{}_Upwork_select".format(local_idx)] == "Yes"
            if not selected:
                continue

            example_id = "{}-{}".format(idx, local_idx)
            wiki_url = row["wiki_url{}".format(local_idx)]
            query = row["Q{}_Rephrase".format(local_idx)]
            answer = row["phrase"]
            texts = tokenizer.tokenize(wiki_article_dict[wiki_url])
            full_context = " ".join(texts).replace("\n", " ")

            # Create short_context by extending N sentences before and M sentences after the sentence containing the answer phrase
            N = random.randint(0, short_context_sentences - 1)
            start_idx = sent_idx - N if sent_idx > N else 0
            end_idx = sent_idx + (short_context_sentences - N) + 1
            short_context = texts[start_idx: end_idx]

            # Exclude all cases where a query can be found in a paragraph by exact match
            if query.lower() not in short_context.lower():
                answer_start = short_context.find(answer) if answer in short_context else short_context.lower().find(answer)
                answer = short_context[answer_start:answer_start+len(answer)] if answer_start != -1 else ""
                list_short_examples.append([example_id, short_context, query, answer, answer_start, wiki_url])

            if query.lower() not in full_context.lower():
                answer_start = full_context.find(answer) if answer in full_context else full_context.lower().find(answer)
                answer = full_context[answer_start:answer_start+len(answer)] if answer_start != -1 else ""
                list_full_examples.append([example_id, full_context, query, answer, answer_start, wiki_url])

            if query.lower() not in short_context.lower() and query.lower() not in full_context.lower():
                list_shared_ids.append(example_id)

    random.seed(42)
    random.shuffle(list_shared_ids)

    # Generate 3 splits: dev 3K, test 5K, train: remaining
    test_ids = random.sample(list_shared_ids, k=5000)
    list_shared_ids = [id for id in list_shared_ids if id not in test_ids]

    dev_ids = random.sample(list_shared_ids, k=3000)
    train_ids_short = [example[0] for example in list_short_examples if example[0] not in dev_ids + test_ids]
    train_ids_full = [example[0] for example in list_full_examples if example[0] not in dev_ids + test_ids]

    generate_csv_files(list_short_examples, train_ids_short, dev_ids, test_ids, setting="PUSS_short")
    generate_csv_files(list_full_examples, train_ids_full, dev_ids, test_ids, setting="PUSS_full")


def analyze_generated_data():
    def extract_pairs(dataset):
        examples = []

        for example in dataset["data"]:
            paragraph = example["paragraphs"][0]
            context = paragraph["context"]
            query = paragraph["qas"][0]["question"]
            answer = paragraph["qas"][0]["answers"][0]["text"]
            examples.append([context, query, answer])

        return examples

    version = "11_sentences"

    with open("../datasets/PhraseSimilarity/PhraseRetrieval/" + "phrase_similarity_v1_{}_SQuAD_format_train.json".format(version), "r", encoding="utf-8") as input_file:
        train_dataset = json.load(input_file)
    with open("../datasets/PhraseSimilarity/PhraseRetrieval/" + "phrase_similarity_v1_{}_SQuAD_format_dev.json".format(version), "r", encoding="utf-8") as input_file:
        valid_dataset = json.load(input_file)
    with open("../datasets/PhraseSimilarity/PhraseRetrieval/" + "phrase_similarity_v1_{}_SQuAD_format_test.json".format(version), "r", encoding="utf-8") as input_file:
        test_dataset = json.load(input_file)

    train_examples = extract_pairs(train_dataset)
    valid_examples = extract_pairs(valid_dataset)
    test_examples = extract_pairs(test_dataset)

    print_unique_statistics(train_examples)
    print_unique_statistics(valid_examples)
    print_unique_statistics(test_examples)
    print_unique_statistics(train_examples + valid_examples + test_examples)

    print_overlapping_statistics(train_examples, valid_examples)
    print_overlapping_statistics(train_examples, test_examples)


def print_unique_statistics(examples):
    contexts = [example[0] for example in examples]
    queries = [example[1] for example in examples]
    answers = [example[2] for example in examples]

    print("\n# unique contexts = {}/{}".format(len(set(contexts)), len(contexts)))
    print("# unique queries = {}/{}".format(len(set(queries)), len(queries)))
    print("# unique answers = {}/{}".format(len(set(answers)), len(answers)))
def print_overlapping_statistics(examples1, examples2):
    contexts1 = [example[0] for example in examples1]
    queries1 = [example[1] for example in examples1]
    answers1 = [example[2] for example in examples1]

    contexts2 = [example[0] for example in examples2]
    queries2 = [example[1] for example in examples2]
    answers2 = [example[2] for example in examples2]

    unique_overlapped_contexts = list(set(contexts1).intersection(set(contexts2)))
    unique_overlapped_queries = list(set(queries1).intersection(set(queries2)))
    unique_overlapped_answers = list(set(answers1).intersection(set(answers2)))

    overlapped_contexts = [context for context in contexts2 if context in unique_overlapped_contexts]
    overlapped_queries = [query for query in queries2 if query in unique_overlapped_queries]
    overlapped_answers = [answer for answer in answers2 if answer in unique_overlapped_answers]

    print("\nOverlapping contexts = {}/{}".format(len(unique_overlapped_contexts), len(overlapped_contexts)))
    print("Overlapping queries = {}/{}".format(len(unique_overlapped_queries), len(overlapped_queries)))
    print("Overlapping answers = {}/{}".format(len(unique_overlapped_answers), len(overlapped_answers)))


if __name__ == '__main__':

    # generate_PS_dataset_using_SQuAD_format(context_levels=[-1, 5])     # [-1, 0, 1, 2, 5]
    # analyze_PS_SQuAD_format_data()

    generate_train_dev_sets_for_PhraseBERT()







