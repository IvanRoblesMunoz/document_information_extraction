# mkdir data/interim
python src/data/wikipedia/parse_wikipedia.py
python src/data/wikipedia/wiki_is_redirect.py.
python src/characterisation/calculate_novelty.py
python src/characterisation/calculate_semantic_similarity.py 

