mkdir -p data/preparation && cd "$_"

curl -X GET https://auburn.box.com/shared/static/1kq59e9wes4nnd38np4hr7p2f8ays046.pickle -L >> amt_data_sorted_by_domain_new_full_latest.csv
curl -X GET https://auburn.box.com/shared/static/anhdkizks1ut54mqxtxfc5hzqo6hiam7.pickle -L >> ambiguous_candidates_-0.3_0.2_5_25_32_3.pickle
curl -X GET https://auburn.box.com/shared/static/zdmxs14wt9hvd1ye4zqblq07ds26al1f.pickle -L >> wiki_categories_dict_cache.pickle
curl -X GET https://auburn.box.com/shared/static/0faptyqotsjwhp7jgbfta209f02dqc8p.pickle -L >> wiki_objects_filtered.pickle
curl -X GET https://auburn.box.com/shared/static/mm8rsmua1cnhlnia4uk73na76zg0usw7.pickle -L >> unique_wiki_url_list.pickle
curl -X GET https://auburn.box.com/shared/static/cwb1z0qmnx518du01ra0vmgndhq7cjry.pickle -L >> unique_wiki_urls_with_categories.pickle

mv wiki_objects_filtered.pickle ../wiki/wikiextractor/

echo "Complete!"


