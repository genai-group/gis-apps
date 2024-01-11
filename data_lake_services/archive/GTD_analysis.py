#!/usr/bin/python

#%%
from app.api.config.init import *

#%%
import os, json, re, sys
import openpyxl
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Union, Any, Optional, Coroutine, Callable, Awaitable, Iterable, AsyncIterable, TypeVar, Generic

# Python charting library
import matplotlib.pyplot as plt

#%%
# import excel file using pandas
def import_excel(file_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

#%%
# import csv file using pandas
def import_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

#%%
# import json file using pandas
data = import_excel("/Users/mattpoulton/Downloads/globalterrorismdb_2021Jan-June_1222dist.xlsx", "Sheet1")

# %%
def get_df_chunks(df: pd.DataFrame, chunk_size: int = 100) -> List[pd.DataFrame]:
    """
    Chunk a pandas dataframe into smaller dataframes

    Args:
        df (pd.DataFrame): pandas dataframe
        chunk_size (int): size of chunks

    Returns:
        List[pd.DataFrame]: list of pandas dataframes

    """
    # assertions
    assert isinstance(df, pd.DataFrame), "df must be a pandas dataframe"
    assert isinstance(chunk_size, int), "chunk_size must be an integer"
    assert chunk_size > 0, "chunk_size must be greater than 0"

    # Chunk the dataframe
    try:
        return np.array_split(df, chunk_size)
    except Exception as e:
        print(f"Errors getting dataframe chunks: {e}")
        return []

# Convert dataframe to list of dictionaries
def df_to_dict_list(df: pd.DataFrame) -> List[Dict]:
    """
    Convert a pandas dataframe to a list of dictionaries

    Args:
        df (pd.DataFrame): pandas dataframe

    Returns:
        List[Dict]: list of dictionaries

    """
    # assertions
    assert isinstance(df, pd.DataFrame), "df must be a pandas dataframe"

    # convert dataframe to list of dictionaries
    try:
        return df.to_dict('records')
    except Exception as e:
        print(f"Errors converting dataframe to list of dictionaries: {e}")
        return []

# determine if object ia a list of dicts or a pandas dataframe. Convert to a list of objects and then chunk to specified list of chunk size
def get_chunks(obj: Union[List[Dict], pd.DataFrame], chunk_size: int = 100) -> List[List[Dict]]:
    """
    Determine if object ia a list of dicts or a pandas dataframe. Convert to a list of objects and then chunk to specified list of chunk size

    Args:
        obj (Union[List[Dict], pd.DataFrame]): list of dictionaries or pandas dataframe
        chunk_size (int): size of chunks

    Returns:
        List[List[Dict]]: list of list of dictionaries

    """
    # assertions
    assert isinstance(obj, (list, pd.DataFrame)), "obj must be a list of dictionaries or a pandas dataframe"
    assert isinstance(chunk_size, int), "chunk_size must be an integer"
    assert chunk_size > 0, "chunk_size must be greater than 0"

    # convert to list of dictionaries
    if isinstance(obj, pd.DataFrame):
        obj = df_to_dict_list(obj)

    # chunk the list of dictionaries
    try:
        return np.array_split(obj, chunk_size)
    except Exception as e:
        print(f"Errors getting list of dictionaries chunks: {e}")
        return []

# Analysis of the data

# Terrorist locations across the world:

#%%
fig = px.scatter_geo(data,lat='latitude',lon='longitude', hover_name="provstate")
fig.update_layout(title = 'World map', title_x=0.5)


# t-SNE visualization of topics in open ended responses

#%%

import pandas as pd
import spacy
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Spacy
import spacy
nlp = spacy.load('en_core_web_lg')

# functions

def preprocess_text(text):
    """
    Tokenizes and lemmatizes the text using spacy, removing stopwords and punctuation.
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def prepare_corpus(data):
    """
    Prepares the corpus from the preprocessed data.
    """
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
    return dictionary, corpus

def perform_lda(corpus, dictionary, num_topics=5):
    """
    Performs LDA analysis and returns the model.
    """
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model

# Load your data into a pandas DataFrame

# Preprocess the text data
df = data

preprocessed_data = df['summary'].apply(preprocess_text)

# Prepare corpus
dictionary, corpus = prepare_corpus(preprocessed_data)

# Perform LDA
lda_model = perform_lda(corpus, dictionary)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=7, angle=.99, init='pca')

