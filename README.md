# Document information extraction
---

# Overview
This repository contains the following sections.
## TODO
Fill in sections here


## 1. Wikipedia dump processing

 In this section, we will process the whole of the latest english wikipedia dump [enwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/enwiki/latest/). 

To parse the wikipedia dump, we will need to move the downloaded data to the data/raw folder. Then we will cd into the repository directory and run the python script that parses the wikipedia corpus.

```
mv enwiki-latest-pages-articles.xml.bz2 data/raw/enwiki-latest-pages-articles.xml.bz2
cd document_information_extraction
python src/data/wikipedia/parse_wikipedia.py 
```

This will read the dump, clean it and save it in an SQL database  in the following path ~/document_information_extraction/data/interim/wiki_db_dumps.db. This can take quite long, it takes ~1h15mins in a 8 core 16 thread machine.

This database will contain two tables.

**wiki_articles**: This contains the wikipedia articles indexed by a page id, the summary is stored as Text in one of the columns, and the body of the text is contained as a list of the sections. The section titles are also stored as a pickled list of titles. Only H2 levels are separated, H3+ are ignored.
```
pageid = int
section_title = pickle.dump(["Heading first section", ..., "Heading last section"])
summary = "Summary text"
body_sections = pickle.dump(["Body first section", ..., "Body last section"])
section_word_count = pickle.dump(["number of words in first section", ..., "number of words in last section"])
```

|pageid|section_title|summary|body_sections|section_word_count|
|------|-------------|-------|-------------|------------------|
|Integer primary key|pickled python list of strings| Text |pickled python  list of strings | pickled python list of int|

**article_level_info**: This contains a pageid key as well as the article title, summary word count and body workd count

|pageid|title|summary_word_count|body_word_count|
|------|-----|------------------|------------------|
|Integer primary key|Text| Integer | Integer |

For a more in depth look into what the final SQL database looks like, check this [notebook](document_information_extraction/notebooks/EDA/01.evaluate-token-distribution.ipynb)

## 2. Evaluate and select best articles for summarisation