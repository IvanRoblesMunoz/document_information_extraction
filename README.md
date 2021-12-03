# Document information extraction
---

## 1. Wikipedia dump processing
 In this section, we will process the whole of the latest english wikipedia dump [enwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/enwiki/latest/). 

To parse the wikipedia dump, we will need to move the downloaded data to the data/raw folder. Then we will cd into the repository directory and run the python script that parses the wikipedia corpus.

```
cd document_information_extraction
# To download data and move to the appropiate folder
shell_comands/01_data_download.sh

# To create the databases
shell_comands/02_make_wiki_database.sh
```

This will read the dump, clean it and save it in an SQL database  in the following path ~/document_information_extraction/data/interim/wiki_db_dumps.db.

It will also create a few other tables used to assess document summarisation suitability, including:
- Semantic similarity between summary and document
- Summary novelty
- Redirect flags    

For a more in depth look into what the final main SQL database looks like, check this [notebook](notebooks/EDA/01.evaluate-token-distribution.ipynb)
And for a datamodel, check this [script](src/data/wikipedia/wiki_data_base.py)


## 2. Pageviews

Due to the size of the database, we will want to find the popularity of articles to then subset the data to keep only the most relevant articles. For example when retrieving relevant passages.

To make the table which contains this data, you can run the following command. This will call the pageviews api to get the data and then store the data.
```
# Collect and store pagviews statistics
shell_commands/03_run_pageviews.sh
```

## 3. Passage embeddings
Finally, we will create passage embeddings, both using BM25 (using sqlite) as well as facebook Dense Pasage Retriever model (queried from faiss). The final database will take around 120GB.

```
# To create passage embeddings
shell_comands/04_make_wiki_database.sh
```

## 4. Open Domain Question Answering
As it is the final embeddings database can be used to query the most relevant passages from all of wikipedia. each query takes less than 1 second. 

A demo can be found in the following [notebook](notebooks/demo_reader_retriver/01_sample_questions.ipynb). 
The following class can be imported and used to query, as long as the databases specified above have been created.

```
from src.retriever.query_retriever import WikiODQARetriever
wiki_odqa = WikiODQARetriever()

query = "What is Data Science ?"
responses = wiki_odqa.retrieve_passage_from_query(query, 5)
for res in responses:
    print("-" * 25)
    for idx, cnt in res.items():
        print(idx, ":", cnt)
    print("-" * 25)
```
