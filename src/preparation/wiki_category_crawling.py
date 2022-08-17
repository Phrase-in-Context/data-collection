import requests
import pickle

from tqdm import tqdm


# Pre-Step 5: Prepare categories for domain embeddings
DATA_PATH = "../../data/"


if __name__ == '__main__':

    # FOR LOADING WIKI URLs ONLY
    with open("unique_wiki_url_list.pickle", "rb") as pickle_file:
        print("Loading Wiki URL list from pickle file...")
        unique_wiki_url_list = pickle.load(pickle_file)

    categories_cache_api = {}
    session = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"

    page_ids = [url.split("=")[-1] for url in unique_wiki_url_list]
    page_size_query = 50    # Maximum limit allowed by Wiki API
    n_batches = int(len(page_ids) / page_size_query) + 1

    count_empty_pages = 0
    for i in tqdm(range(n_batches), total=n_batches):
        sub_page_ids = page_ids[i*page_size_query: (i+1)*page_size_query]
        sub_page_ids_query = "|".join(sub_page_ids)
        PARAMS = {
            "action": "query",
            "format": "json",
            "prop": "categories",
            "clshow": "!hidden",
            "cllimit": "500",
            "pageids": str(sub_page_ids_query)
        }

        response = session.get(url=URL, params=PARAMS)
        data = response.json()
        pages = data["query"]["pages"]

        for article in pages.values():
            if 'categories' not in article:
                count_empty_pages += 1
                categories_cache_api["https://en.wikipedia.org/wiki?curid={}".format(article["pageid"])] = []
                continue

            categories = [cat["title"].split("Category:")[-1] for cat in article['categories']]
            categories_cache_api["https://en.wikipedia.org/wiki?curid={}".format(article["pageid"])] = categories

    print("Number of empty Wiki pages = {}".format(count_empty_pages))
    print("Dumping all UNIQUE Wiki URL to pickle file...")

    wiki_articles_pickle_fp = DATA_PATH + "preparation/sub_wiki_url_with_categories.pickle"
    with open(wiki_articles_pickle_fp, "wb") as pickle_file:
        pickle.dump(categories_cache_api, pickle_file)


