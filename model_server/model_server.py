# -*- coding: utf-8 -*-
"""
Modified https://github.com/chain-ml/tmls-2023-material/blob/main/ChainML_TMLS2023.ipynb
"""

"""### Import dependencies"""

# Python Standard Library imports
from string import Template
import os
from heapq import nlargest

# Third-Party Library imports
import json
import requests
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from googleapiclient.discovery import build
from pprint import pprint

# Sentence Transformers imports
from sentence_transformers import SentenceTransformer, CrossEncoder

# Transformers import
from transformers import AutoTokenizer

# OpenAI import
import openai

# GoogleNews import
from GoogleNews import GoogleNews

# WikipediaAPI import
import wikipediaapi

# Tiktoken import
import tiktoken

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Load the environment variable and set OpenAI API key
load_dotenv('config.env', override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

"""# (12) Making Calls to GPT-4"""

# Function for getting a single chat completion
def get_completion(message, system_prompt='', model='gpt-4'):
  completion = openai.ChatCompletion.create(
    model=model,
    temperature=0,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
  )
  return completion["choices"][0]["message"]["content"]

# TEST
# user_message = "How do I connect to Uniswap?"
# print(get_completion(user_message, model="gpt-4"))

"""# Context-Building Skills

## Data Preprocessing

A standard approach to doing text-based search involves the transformation of sequences of text (chunks) into embedding vectors. These are high-dimensional represenations of text sequences produced by a pre-trained language model. Many of these are available in the `sentence_transformers` Python package.

After converting text chunks to embedding vectors, we'll be able to match them to a user (or AI-generated) query by computing a distance metric between each of the query-chunk embedding pairs. Even with thousands of chunks, this process can be computed almost instantly and will enable us to quickly identify parts of the source documents that are related to the user's question.
"""

import hashlib

from transformers import AutoTokenizer

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

def tokenizer_len(text):
    tokens = embedding_tokenizer.encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=64,
    length_function=tokenizer_len
)

def extract_pages(file_path, lower_page_num, upper_page_num):

    """Function to extract text from PDF file"""

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split(text_splitter=text_splitter)
    filtered_pages = [page for page in pages if page.metadata['page'] in list(range(lower_page_num, upper_page_num))]

    return filtered_pages

def format_chunk(text,
                 metadata,
                 chunk_number):

    """Function to format every text chunk for entry into vector database"""

    chunk = {}

    # PyPDF indexes pages starting from 0
    page_number = metadata['page'] + 1
    source = metadata['source']

    # For creating unique id of text chunk
    m = hashlib.md5()
    m.update(f"{source}-{chunk_number}".encode('utf-8'))
    uid = m.hexdigest()[:12]

    chunk['id'] = uid
    chunk['text'] = text
    chunk['metadata'] = {}
    chunk['metadata']['page_number'] = page_number
    chunk['metadata']['source'] = source

    return chunk

def create_dataset(pdf_files,
                   save_file_path,
                   ):

    """Function to extract text from PDF files, format for entry into vector database and save to disk."""

    chunked_documents = []

    for file in pdf_files:
        pages = extract_pages(file_path=file['path'],
                              lower_page_num=file['lower_page_num'],
                              upper_page_num=file['upper_page_num'])

        for i, page in enumerate(pages):
            chunk = format_chunk(text=page.page_content,
                                 metadata=page.metadata,
                                 chunk_number=i)
            chunked_documents.append(chunk)

    with open(save_file_path, 'w') as f:
        for doc in chunked_documents:
            f.write(json.dumps(doc) + '\n')

uni_files = [
    {
        "path": "data/uniswap_docs_pdf_combined.pdf",
        "lower_page_num": 0,
        "upper_page_num": 1036
    }
]

UNI_DATA_PATH = "data/uni_dataset.jsonl"


# Create Uniswap dataset
create_dataset(uni_files, UNI_DATA_PATH)

"""## Creating a database of text embeddings

We are using Chroma, an open-source database purpose built for AI embedding vectors and operations. This allows us to build a prototype embedding database that can scale to a far greater number of source documents.
"""

import chromadb
from chromadb.config import Settings

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

tokenizer = tiktoken.get_encoding("cl100k_base")


uni_data = []
with open('data/uni_dataset.jsonl', 'r') as f:
    for line in f:
        uni_data.append(json.loads(line))

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb/"
))


uni_collection = client.get_or_create_collection("UNI", metadata={"hnsw:space": "cosine"})

batch_size = 128

def populate_db(db_collection,
                data,
                embedding_model,
                batch_size):


    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        texts = []
        ids = []
        metadatas = []

        for doc in batch:

            doc_id = doc["id"]
            text = doc["text"]
            metadata = doc["metadata"]

            ids.append(doc_id)
            metadatas.append(metadata)
            texts.append(text)

        # Generate embeddings for the entire batch
        embeddings = embedding_model.encode(
            texts, convert_to_tensor=False
        ).tolist()

        # Store the batch to the database
        db_collection.add(
            documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )


# Populate database with Uniswap documents
# populate_db(uni_collection, uni_data, embedding_model, batch_size)
print(len(uni_data))
print(uni_collection.count())

# Persist database to disk
client.persist()

"""## Base Functions for Skills

### Retrieval Functions
"""

def retrieve_docs(query,
                  k,
                  collection):

    """Function to retrieve top K documents from the database based on similarity to the query."""

    # Calculate the embedding for the query
    embedded_query = embedding_model.encode(query, convert_to_tensor=False).tolist()

    # Retrieve documenents from the database
    output = collection.query(
        query_embeddings=[embedded_query],
        n_results=k
    )

    return output


def retrieve_news(query,
                  k):

    """Retrieve K news articles from Google News"""

    google_news = GoogleNews(period='30d')
    google_news.enableException(True)

    results = []
    google_news.search(f"{query}".replace(' ', '+'))

    page_num = 1

    while len(results) < k:

        try:
            google_news.get_page(page_num)
            google_news_result = google_news.results()

            if len(google_news_result) == 0:
                break
            results.extend(google_news_result)
            page_num += 1
        except:
            break

    google_news.clear()

    return results

def unique_elements(input):

    seen = {}
    for item in input:
        if item.get("link") not in seen:
            seen[item.get("link")] = item

    return list(seen.values())

"""### Utility Functions"""

def rank_documents(sentence_pairs, num_results):

    """Rank the retrieved documents based on their relevance to user query"""

    rankings = cross_encoder_model.predict(sentence_pairs)
    top_ranked_idx = rankings.argsort()[-num_results:][::-1]

    return top_ranked_idx


def create_context(documents, token_limit):

    """Convert the results of the retrieved documents into a context for the LLM prompt."""

    context = ""
    num_tokens = 0

    for doc in documents:
        new_num_tokens = num_tokens + len(tokenizer.encode(doc))
        if new_num_tokens <= token_limit:
            context += f"{doc}\n\n"
            num_tokens = new_num_tokens
        else:
            break

    return context

"""### Querying Functions"""

def query_database(collection,
                   query,
                   num_retrieved_docs,
                   num_ranked_docs,
                   token_limit):

    # Retrieve documents from database
    retrieved_docs = retrieve_docs(query, num_retrieved_docs, collection)

    # Create sentence pairs for ranking
    documents = retrieved_docs['documents'][0]
    sentence_pairs = [[query, doc] for doc in documents]

    # Get indices of top ranked documents
    top_ranked_idx = rank_documents(sentence_pairs, num_ranked_docs)

    # Create updated document set
    ranked_results = {}
    ranked_results['documents'] = [[retrieved_docs['documents'][0][i] for i in top_ranked_idx]]
    ranked_documents = [retrieved_docs['documents'][0][i] for i in top_ranked_idx]

    # Create the context
    context = create_context(ranked_documents, token_limit)

    return context


def query_google_news(query,
                      num_retrieved_docs,
                      num_ranked_docs,
                      token_limit):

    # Retrive Google News results
    retrieved_news = retrieve_news(query, num_retrieved_docs)
    # Filter out duplicates
    retrieved_news = unique_elements(retrieved_news)

    # Create sentence pairs for ranking
    documents = []
    for result in retrieved_news:
        text = f"{result['title']} + {result['desc']}"
        documents.append(text)

    sentence_pairs = [[query, doc] for doc in documents]

    # Get indices of top ranked documents
    top_ranked_idx = rank_documents(sentence_pairs, num_ranked_docs)

    # Create updated document set
    ranked_documents = [documents[i] for i in top_ranked_idx]

    # Create the context
    context = create_context(ranked_documents, token_limit)

    return context

"""## Agent Skill Functions"""

# Setting constants
NUM_RETRIEVED_DB_DOCS = 100
NUM_RETRIEVED_NEWS_DOCS = 20
NUM_RANKED_DOCS = 2
CONTEXT_TOKEN_LIMIT = 4000

def uni_docs_skill(query):

    """Skill to create a context for the LLM by retrieving relevant documents from the Uniswap document database"""
    return query_database(collection=uni_collection,
                          query=query,
                          num_retrieved_docs=NUM_RETRIEVED_DB_DOCS,
                          num_ranked_docs=NUM_RANKED_DOCS,
                          token_limit=CONTEXT_TOKEN_LIMIT)

def google_news_skill(query):

    """Skill to create a context for the LLM by retrieving relevant news articles from Google News"""
    return query_google_news(query=query,
                             num_retrieved_docs=NUM_RETRIEVED_NEWS_DOCS,
                             num_ranked_docs=NUM_RANKED_DOCS,
                             token_limit=CONTEXT_TOKEN_LIMIT)

"""### (15) Skill Testing Questions"""

# print(google_news_skill("how do i connect to uniswap?"))

# print(uni_docs_skill("how do i connect to uniswap?"))

"""# Using GPT-4 For Skill Selection

## (16) Implementing A Basic Control Strategy
"""

controller_system_prompt = """
You are the control module for an expert Web3 Developer Evangelist who is responsible
for assisting human users with understanding and interpreting documentation
and other sources of data about Web3 projects.
Your role is to determine which of your skills, called chains, are most likely
to help the Web3 Developer Evangelist with answering the user question.
"""

controller_prompt_template = Template("""
### Goal
Determine which of the following chains, formatted as a Python
dictionary {chain_name: chain_description}, are relevant to the user message,
and reformulate the user message into a brief search query.

### Chains
$chains

### Instructions
- Generate a response for each of the provided chains
- In each response, reformulate the user message into a brief and relevant search query
- For chains that are irrelevant, respond with "N/A" for the reformulated query
- Rank your responses from most to least relevant
- Format each response precisely as: rank;chain_name;reformulated search query

### Examples
$few_shot_examples

############
User message:
$user_message

Response:
""")

chains = {
    "uni_investor_material": "Retrieve information from Uniswap's official documentation that contains important information about how Uniswap works and how it can be used.",
    "google_news_search": "Search Google News for recent information related to the user message. Be sure to mention the relevant company name in the reformulated search query."
}


few_shot_examples = "\n***\n".join(["""
User message:
Does Uniswap use the contract factory design pattern?

Response:
1;uni_investor_material;factory contract
2;google_news_search;Uniswap factory contract
"""
])

# Preview the prompt after filling in variables

# print(controller_prompt_template.substitute(
#     chains=chains,
#     few_shot_examples = few_shot_examples,
#     user_message=''
# ))

"""## Controller Test Run"""

# Use GPT to generate a response to the control request

user_message = "How do I connect to Uniswap"

message = controller_prompt_template.substitute(
    chains=chains,
    few_shot_examples = few_shot_examples,
    user_message=user_message,
)

response = get_completion(message, controller_system_prompt, model='gpt-4')
print(response)

# Parse the response
def parse_top_response(response):
  try:
    responses = response.split('\n')
    rank, chain_name, query = responses[0].split(';')
    return chain_name, query
  except Exception as e:
    print("Failed to parse response: {response}")
    return None

# Map chain names to callable skill functions
skills_map = {
    "uni_investor_material": uni_docs_skill,
    "google_news_search": google_news_skill
}

# Check the parsed top response
skill, query = parse_top_response(response)
print(f"{skill}, {query}")

# Apply the top-ranked skill with the generated query
print(skills_map[skill](query))

"""## (17) Calling GPT With Context"""

web3_devrel_system_prompt = """
You are the control module for an expert Web3 Developer Evangelist who is responsible
for assisting human users with understanding and interpreting documentation
and other sources of data about Web3 projects.
Relevant information to users' queries will be provided as input. If
it is useful, use this information to help answer the question.
"""

web3_devrel_prompt_template = Template("""
### Goal
Respond to the user's message, using the following additional information, if it is useful.

### User Message
$user_message

### Context
$context

### Instructions
- Respond to the User Message to the best of your ability.
- Only provide responses that are factually correct.
- If you don't know the answer to a question, say this in your response.

Response:
""")

context = skills_map[skill](query)

"""### Checking Outputs"""

# Check the prompt with supplied context
print(web3_devrel_prompt_template.substitute(
    user_message=user_message,
    context=context
))

"""## Web3 DevRel Test Run"""

message = web3_devrel_prompt_template.substitute(
    user_message=user_message,
    context=context
)

response = get_completion(message, web3_devrel_system_prompt, model='gpt-4')
pprint(response)

"""# Self-Assessment Via Fact-Checking

We have four steps for evaluating factuality of a statement:
1. Generate Knowledge base (KBs) search queries
2. Search and retrieve documents from KBs using generated queries
3. Rank and extract the most relevant passages from the retrieved documents
4. Generate a True/False response for the given statement and fact evidence passage.

System returns  `True` if we could find enough evidence in KBs to support the fact. Else `False`.

Let's continue with the same example user message and AI-generated output.
"""

# Recall our working example
print("User Message:")
print(user_message, '\n')
print("AI Response:")
pprint(response)

"""Our next goal is to validate whether we have supporting evidence for this response.

## Generate Search Queries for Fact Checking

### Setting Constants
"""

NUM_QUERIES = 3 #number of search queries to generate
NUM_WIKI_ARTICLES = 2 #max number of wiki docs to scrape for each query
NUM_GOOGLE_NEWS_ARTICLES = 3 #max number of google news docs to scrape for each query
NUM_GOOGLE_SEARCH_ARTICLES = 3 #max number of google search docs to scrape for each query
DEFAULT_MODEL = "gpt-4"

NUM_RETRIEVED_DB_DOCS = 100
NUM_RETRIEVED_NEWS_DOCS = 20
NUM_RANKED_DOCS = 10
CONTEXT_TOKEN_LIMIT = 4000

fc_queries_system_prompt = """
You are the query generation mechanism for a fact-checking agent.
Your role is to generate search queries that will be used to fetch
articles that can help with verifying facts in an AI-generated response.
Your response should only be a list of queries separated by '\n'.
e.g.: `foo\n bar\n baz`
"""

fc_queries_prompt_template = Template("""
### Goal
Generate fact-checking queries for:

$ai_response

### Instructions
- Generate the response as a list of queries separated by '\n'.

### Examples
foo\n bar\n baz
query1\n query2\n query3

""")

# def generate_queries(ai_response, num_queries=NUM_QUERIES, model="gpt-3.5-turbo", temprature=0):
def generate_queries(ai_response, num_queries=NUM_QUERIES, model="gpt-4", temprature=0):

    fc_message = fc_queries_prompt_template.substitute(ai_response=ai_response)
    fc_queries_response = get_completion(fc_message, fc_queries_system_prompt)

    queries = fc_queries_response.strip().split("\n")[:num_queries]
    for i in range(len(queries)):
      queries[i] = queries[i].strip()
      if queries[i][:2] == "- ":
          queries[i] = queries[i][2:].strip()
    queries = [q for q in queries if q is not None and len(q)]
    return queries

queries = generate_queries(response, num_queries=NUM_QUERIES, model='gpt-4')
pprint(queries)

"""### Utility Functions"""

def chunk_text(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

    def tokenizer_len(text):
        tokens = tokenizer.encode(text, max_length=512, truncation=False)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=128,
            chunk_overlap=64,
            length_function=tokenizer_len
    )
    return text_splitter.split_text(text)


def search_wiki(queries, n_max_articles=NUM_WIKI_ARTICLES):
    documents = []
    wiki_api_url = "https://en.wikipedia.org/w/api.php"
    for query in queries:
        #search documents
        search_response = requests.get(
            wiki_api_url,
            params={
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": query
                }
        ).json()
        page_titles = [q['title'] for q in search_response['query']['search'][:n_max_articles]]

        #fetch documents
        wiki_wiki = wikipediaapi.Wikipedia(
                    language='en',
                    extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        for page_title in page_titles:
            try:
                document = wiki_wiki.page(page_title).text
                url = f"https://en.wikipedia.org/wiki/{page_title}"
            except:
                continue

            text_chunks = chunk_text(document)
            for chunk in text_chunks:
                documents.append({
                    "text": chunk,
                    "metadata": {"source": url},
                })
    return documents

def remove_duplicate_docs(fact_documents):
    texts = set()
    unique_fact_documents = []
    for doc in fact_documents:
        if doc['text'] in texts:
            continue
        unique_fact_documents.append(doc)
        texts.add(doc['text'])
    return unique_fact_documents

def search_google(queries, n=NUM_GOOGLE_SEARCH_ARTICLES):
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("CX")
    google_search_obj = build("customsearch", "v1", developerKey=api_key)
    search_docs = []
    for query in queries:
        response =  google_search_obj.cse().list(q=query, cx=cx, num=n).execute() #load from env
        data = response.get("items", [])
        for d in data[:n]:
            search_docs.append({
                "text": f"{d['title']}\n{d['snippet']}",
                "metadata": {"source": d['link']},
            })
    return search_docs

def retrieve_fact_docs(
        queries,
        n_news_docs=NUM_GOOGLE_NEWS_ARTICLES,
        n_wiki_docs=NUM_WIKI_ARTICLES,
        n_google_search_docs=NUM_GOOGLE_SEARCH_ARTICLES):
    #Google news docs
    news_docs = []
    for query in queries:
        try:
            news_articles = retrieve_news(query, n_news_docs)
        except:
            continue
        for article in news_articles:
            news_docs.append({
                "text": f"{article['title']}\n{article['desc']}",
                "metadata": {"source": article['link']},
            })

    #Wiki docs
    wiki_docs = search_wiki(queries, n_wiki_docs)

    #Google search docs
    google_search_docs = search_google(queries, n_google_search_docs)

    # Combine all docs and remove duplicates
    fact_documents = news_docs + wiki_docs + google_search_docs
    fact_documents = [{'id': str(i), **doc} for i, doc in enumerate(fact_documents)]
    fact_documents = remove_duplicate_docs(fact_documents)

    return fact_documents

"""## Retrieve Documents Related to Generated Queries"""

fact_documents = retrieve_fact_docs(queries)

# # Print an example
# pprint(fact_documents[0])

"""## Find The Most Related Search Results

- Extract the most-informative sections from the retrieved documents
- Use the sentence transformer to encode chunks of each document as overlapping chunk vectors.
- Encode the `response` using the same transformer, call this the "query" vector
- Find the search result chunks with the highest cosine similarity to the previously generated response
"""

def extract_most_relevant_fact_passage(test_response, fact_documents, max_passage_tokens=300):
    fact_evidence_collection = client.get_or_create_collection(f"FEC", metadata={"hnsw:space": "cosine"})
    populate_db(fact_evidence_collection, fact_documents, embedding_model, batch_size)

    relevant_fact_passage = query_database(
        collection=fact_evidence_collection,
        query=test_response,
        num_retrieved_docs=5,
        num_ranked_docs=NUM_RANKED_DOCS,
        token_limit=CONTEXT_TOKEN_LIMIT
    )
    relevant_fact_passage = tokenizer.decode(tokenizer.encode(relevant_fact_passage)[:max_passage_tokens])
    client.delete_collection("FEC")
    return relevant_fact_passage

# relevant_fact_passages = extract_most_relevant_fact_passage(response, fact_documents)

# print(relevant_fact_passages)

"""## (22) Use an Evaluator Agent to Perform Fact Checking

Using the content retrieved via search, determine whether it has contains information or evidence that supports the initial AI-generated response.
"""

fc_system_prompt = "You are a fact evaluator Agent responsible for fact checking AI generated statements with the help of a given related information passage."

fc_message_prompt_template = Template("""
### Goal
Evaluate the factuality of an AI-Generated Response to a User Message using the given Retrieved Context and Related Information.

### Instructions
- First, write out in a step-by-step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.
- Then, on a new line, write "True" if the Related Information supports the AI-Generated Response, otherwise write "False".

### Example Output
There was sufficient Related Information to support the claim.

True

************************

There was not sufficient information to support the claim.

False

### User Message
$user_message

### AI-Generated Response
$ai_response

### Retrieved Context
$context

### Related Information
$passage

### Response
""")

def evaluate_fact(user_message, context, relevant_fact_passage, ai_response):
    message = fc_message_prompt_template.substitute(user_message=user_message, context=context, ai_response=ai_response, passage=relevant_fact_passage)
    fc_compare_response = get_completion(message, fc_system_prompt)
    return fc_compare_response.strip()

# response

# context

# relevant_fact_passages

# print(evaluate_fact(user_message, context, relevant_fact_passages, response))

"""## Fact-Checking Test Run"""

def run_fact_check(user_message, context, ai_response):
  search_queries = generate_queries(ai_response, NUM_QUERIES)
  document_chunks = retrieve_fact_docs(search_queries)
  relevant_fact_passage = extract_most_relevant_fact_passage(ai_response, fact_documents)
  return evaluate_fact(user_message, context, relevant_fact_passage, ai_response)

# response

# fact_checking_response = run_fact_check(user_message, context, response)

# print(fact_checking_response)

"""# Building An End-To-End Assistant

## (23) Collecting All Steps
"""

### Bring everything together so that we can build a single interface for the end user, so we can go directly from
### user message to a fact-checked response.

def web3_devrel_chain(user_message, verbose=False):

  # The User's message to the AI system
  if verbose:
    print(f"User's Message:\n{user_message}\n")

  # Apply the Controller prompt template to build the Controller message
  controller_message = controller_prompt_template.substitute(
      chains=chains,
      few_shot_examples = few_shot_examples,
      user_message=user_message,
  )

  # Get the Controller's response
  controller_response = get_completion(controller_message, controller_system_prompt, model='gpt-4')
  if verbose:
    print(f"Controller's Response:\n{controller_response}\n")

  # Pasre the Controller's response
  skill, query = parse_top_response(controller_response)
  if verbose:
    print(f"Parsed Controller Response:\nskill: {skill}, query: {query}\n")

  # Use the selected skill to get relevant contextual information
  context = skills_map[skill](query)
  if verbose:
    print("Retrieved Contextual Information")
    pprint(context)
    print()

  # Apply the Web3 DevRel prompt template to build the Web3 DevRel message
  web3_devrel_message = web3_devrel_prompt_template.substitute(
      user_message=user_message,
      context=context
  )

  # Get the Web3 DevRel's response
  web3_devrel_response = get_completion(web3_devrel_message, web3_devrel_system_prompt, model='gpt-4')
  if verbose:
    pprint(f"Web3 DevRel's Response:\n{web3_devrel_response}")
    print()

  # Run Fact-Checking and get the response
  fact_checking_response = run_fact_check(user_message, context, web3_devrel_response)
  if verbose:
    print(f"Fact-Checking Result:\n{fact_checking_response}\n")

  return {
      "user_message": user_message,
      "controller_response": controller_response,
      "selected_skill": skill,
      "generated_query": query,
      "retrieved_context": context,
      "web3_devrel_response": web3_devrel_response,
      "fact_checking_response": fact_checking_response,
      "fact_check_passed": True if fact_checking_response.split()[-1] == 'True' else False
  }


# user_message = "How do I connect to Uniswap?"
# web3_devrel_output = web3_devrel_chain(user_message, verbose=True)

def self_assessed_web3_devrel(user_message, verbose=False):
  web3_devrel_output = web3_devrel_chain(user_message, verbose=verbose)
  print(f"{web3_devrel_output['user_message']}\n")
  if web3_devrel_output['fact_check_passed']:
    print("NOTICE: While automated fact-checking was successful, the factual accuracy of AI-generated responses is not guaranteed.\n")
    pprint(web3_devrel_output['web3_devrel_response'])
  else:
    print("WARNING: Supporting evidence was not found, please interpret with caution.\n")
    pprint(web3_devrel_output['web3_devrel_response'])

"""## (24) Testing with additional user messages"""


# self_assessed_web3_devrel("How do i connect to uniswap?", verbose=False)

"""# Comparing with Vanilla GPT-4 Response"""

# vanilla_response = """
# To connect to Uniswap, you will need to follow these steps:
#
# 1. Set up a Web3-enabled wallet: Uniswap is built on the Ethereum blockchain, so you'll need a Web3-enabled wallet to interact with it. Popular options include MetaMask (a browser extension) and Trust Wallet (a mobile app). Install the wallet of your choice and create a new Ethereum wallet if you don't have one already. Be sure to securely back up your wallet's recovery phrase or private key.
#
# 2. Fund your wallet with Ethereum (ETH): Uniswap operates on the Ethereum network, so you'll need some ETH to pay for transaction fees and interact with the platform. Purchase ETH from a reputable exchange and transfer it to your wallet address.
#
# 3. Access the Uniswap interface: Visit the Uniswap website at https://app.uniswap.org/. Ensure that you are on the correct website and exercise caution to avoid phishing attempts. You can also use alternative interfaces like Uniswap.info or UniswapV2, but the official Uniswap interface is recommended.
#
# 4. Connect your wallet: On the Uniswap interface, locate the "Connect Wallet" or similar button. Click on it, and it will prompt you to connect your wallet. If you're using MetaMask or Trust Wallet, a pop-up window should appear automatically, requesting permission to connect.
#
# 5. Confirm connection and permissions: After clicking on the connect button, you'll likely see a pop-up or prompt from your wallet. Review the permissions requested by the Uniswap interface (such as access to your wallet address and balances) and approve them. This step allows the Uniswap interface to interact with your wallet on your behalf.
# 
# 6. Interact with Uniswap: Once your wallet is successfully connected, you can begin using Uniswap. You can swap tokens, provide liquidity to pools, or participate in other features offered by the protocol. Explore the different options available on the interface, and ensure you understand the risks and mechanics of the transactions you wish to perform.
#
# Remember to exercise caution when interacting with decentralized platforms like Uniswap. Verify the URLs and interfaces to avoid phishing attempts, and be mindful of the assets you're interacting with. It's also a good practice to start with smaller transactions until you are comfortable with the process.
# """

# message = "How do I connect to Uniswap?"
# fact_checking_response = run_fact_check(message, context='', ai_response=vanilla_response)
# print(fact_checking_response)

"""### More Queries"""

# self_assessed_web3_devrel("What is the Uniswap Router about?", verbose=True)

"""
FLASK SERVER
"""

from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

"""
{
    "user": ...,
    "prompt": "...",
    "timestamp": ...
}
"""
@app.route("/chat", methods=["POST"])
def chat():
    req_data = request.get_json()

    prompt = req_data["prompt"]
    print(prompt)

    bot_response = web3_devrel_chain(prompt, verbose=False)
    print(bot_response)

    return bot_response

app.run(port=5555, debug=True)

