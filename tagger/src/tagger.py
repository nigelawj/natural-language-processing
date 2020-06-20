# Imports
from elasticsearch import Elasticsearch, ConnectionError, ElasticsearchException
from elasticsearch.helpers import bulk
from elasticsearch.client import IndicesClient

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import sys, os
import argparse
import time
import logging

from nltk.tokenize import word_tokenize
import re, string

# Function Definitions
def handle_exception(exc_type, exc_value, exc_traceback):
    '''
    Function to overwrite sys.excepthook to modify exceptions flow

    Args:
        exc_type (type)
        exc_value (NameError)
        exc_traceback (traceback)
    '''
    # Code block to ignore KeyboardInterrupt exception
    # NOTE: KeyboardInterrupt cannot be captured, likely due to async issues. 
    #       Force raising a KeyboardInterrupt will trigger this 'if' block, but currently this 'if' block has no effect
    if issubclass(exc_type, KeyboardInterrupt):
        rootLogger.critical("Keyboard Interrupted.")
        sys.exit(3)

    # Uncaught exceptions
    rootLogger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def check_time(workStartAM, workEndHr, rootLogger):
    '''
    Sanity-check: Check if tagger is executing during work hours; resume/defer tagging after work hours
    Default work hours: 8 AM - 6 PM

    Args:
        workStartAM (int): Work starting HR - ranges from 0 - 24
        workEndHr (int): Work ending HR - ranges from 0 - 24
        rootLogger (obj): reference of rootLogger object
    
    '''
    workEndPM = workEndHr - 12 # convert to 12hr format

    if (time.localtime().tm_hour in range(workStartAM, workEndHr)):
        rootLogger.error(f"It is past {workStartAM} AM. Tagging should only be resumed at {workEndPM} PM onwards.")
        sys.exit(5)

def connectDB(esIndex, nodes, rootLogger):
    '''
    Function to connect to Elasticsearch DB. Uses default parameters.

    Args:
        esIndex (str): Elasticsearch index of concern
        nodes (list): list of string values of node information; e.g. ['127.0.0.1:9200', '127.0.0.2:9200']
        rootLogger (obj): reference of rootLogger object

    Returns:
        obj: elasticsearch object reference
    '''
    es = Elasticsearch([node for node in nodes])
    try:
        # Get no. of documents
        numDocs = es.count(index=esIndex, body={"query": {"match_all": {}}})['count']
        rootLogger.info(f'Connection successful. Number of documents found: {numDocs}')
        return es
    except ConnectionError:
        rootLogger.error('Error talking to ES DB. Check if DB is started up.')
        sys.exit(500)
    except ElasticsearchException:
        rootLogger.error("Unexpected error:", sys.exc_info()[0])
        sys.exit(500)

def load_stop_words(path):
    '''
    Function to load stopwords into a list. 

    Args:
        path (str): path of stopwords.txt file

    Returns:
        list: returns list of stopwords
    '''
    stop_words = set()

    with open(path, 'r') as f:
        text = f.read()
        stop_words.update(re.split(r'[;,\s\n]\s*', text))
        
    return list(stop_words)

def preprocess_text(doc, stop_words):
    '''
    Function to preprocess documents: convert to lowercase, filter stopwords via a stopwords .txt file, etc.

    Args:
        doc (str): document string to be preprocessed
        stop_words (list): list of stopwords to use

    Returns:
        str: returns preprocessed text
    '''
    # split into words
    tokens = word_tokenize(doc)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    stripped_text = [word for word in stripped if word.isalpha()]

    # filter out stop words
    stripped_text = [w for w in stripped_text if not w in stop_words]

    doc = ' '.join(stripped_text)
    return doc

def perform_LDA(doc, max_features, max_iter, learning_offset, top_words):
    '''
    Function to perform sk-learn LDA

    Args:
        doc (str): document (preprocessed) to perform LDA on

        The following parameters should be passed in via tagger.ps1 or tagger.py args:

            max_features (int): number of features used in LDA
                - default: 1000
            max_iter (int): number of iterations
                - default: 1000
            learning_offset (int): learning offset
                - default: 50
            top_words (int): number of top words to display
                - default: 10

    Returns:
        list: returns list of lists of tags extracted; NOTE: list contains a list of topic-word lists, for our case no_topics=1 
    '''
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    tf = tf_vectorizer.fit_transform([doc]) # fit on 1 document
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Run LDA
    lda = LatentDirichletAllocation(n_components=1, max_iter=max_iter, learning_method='online', learning_offset=learning_offset,random_state=0).fit(tf)

    lda_terms = []
    for topic_idx, topic in enumerate(lda.components_):
        lda_terms.append([tf_feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]])
        
    return lda_terms

def get_actions(res_responses, esIndex, rootLogger, stop_words, max_features, max_iter, learning_offset, top_words):
    '''
    Function to process documents, input in an object containing ES _search results, and output yields an action per document

    Args:
        res_responses (dict): responses obtained by the es.msearch() function
        esIndex (str): Elasticsearch index of concern
        rootLogger (obj): reference of rootLogger object

        The following parameters should be passed in via tagger.ps1 or tagger.py args:

            max_features (int): number of features used in LDA
                - default: 1000
            max_iter (int): number of iterations
                - default: 1000
            learning_offset (int): learning offset
                - default: 50
            top_words (int): number of top words to display
                - default: 10
    '''
    for res in res_responses: # two halves (top and bottom half) of the msearch result
        for i in range(len(res['hits']['hits'])): # for each document
            doc_id = res['hits']['hits'][i]['_id']
            doc = res['hits']['hits'][i]['_source']['content']
            rootLogger.info(f'Processing Document {i}: {doc_id}')

            # If document is empty, return empty tags
            if (doc.replace(' ', '') == ''):
                tags = [['']]
            else:
                processed_text = preprocess_text(doc, stop_words)
                tags = perform_LDA(processed_text, max_features, max_iter, learning_offset, top_words)
                
            rootLogger.info(f'LDA output of Document {i}: {tags[0]}')
            
            yield { # action for each document
                '_op_type': 'update',
                '_index': esIndex,
                '_type': '_doc',
                '_id': doc_id,
                '_source': {
                    'doc': {
                        'tags': tags[0],
                        'lastTagged': round(time.time())
                    },
                }
            }

def execute_es_query(es, esIndex, b, o):
    '''
    Constructs and executes the es.msearch() query

    Args:
        es (obj): elasticsearch object reference
        esIndex (str): Elasticsearch index of concern

        The following parameters should be passed in via tagger.ps1 or tagger.py args:

            b (int): batch size
                - default: 100
            o (int): time in unix epoch seconds, which documents tagged before this time will be flagged for re-tagging
                - default: 0

    Returns:
        dict: returns the result in the form of a json object (dict in python)
    '''
    # Construct req array
    req = []
    req_head = {
        "index": esIndex
    }
    size = int(b/2)

    # Documents tagged prior to specified re-tag timestamp 'o' will be re-tagged; arg passed in via shell script
    # NOTE: Untagged documents will simply be tagged as per normal.
    #       re-tag timestamp will serve as a marker for script to recognise documents with tags that still pending re-tagging
    
    # First Half
    req_body = {
        "size": size,
        "query": {
            "bool": {
                "should": [{
                    # Documents tagged prior to re-tag timestamp specified
                    "range": {
                        "lastTagged": { # Untagged documents without this field will not throw errors
                            "lt": o
                        }
                    }
                }, {
                    # OR documents without tags
                    "bool": {
                        "must_not": {
                            "exists": {
                                "field": "tags"
                            }
                        }
                    }
                }]
            }
        },
        # Sort by lastIndexed field and grab top b/2 documents
        # "sort" will grab top and bottom b documents (Verified); allows tagging recently added documents to prevent long wait time
        # NOTE: Documents might overlap, but if it does, database is so small, overhead is negligible
        "sort" : [{
            "lastIndexed": {
                "order": "desc"
            }
        }],
        "_source": "content" # only content field is required as of now; to be changed if needed in the future
    }
    req.extend([req_head, req_body])

    # Second Half
    # NOTE: Should keep query identical to top half; if many more new documents came after re-tagging was triggered,
    #       the bottom half of query can 'assist' in tagging documents instead of idling if it has no more documents to process
    req_body = {
        "size": size,
        "query": {
            "bool": {
                "should": [{
                    "range": {
                        "lastTagged": {
                            "lt": o
                        }
                    }
                }, {
                    "bool": {
                        "must_not": {
                            "exists": {
                                "field": "tags"
                            }
                        }
                    }
                }]
            }
        },
        "sort" : [{
            "lastIndexed": {
                "order": "asc"
            }
        }],
        "_source": "content"
    }
    req.extend([req_head, req_body])
    res = es.msearch(body=req)

    return res

if __name__ == "__main__":
    # 0a. Initialise logger
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG) # Set logger level

    # fileHandler handles logging to log file
    # NOTE: Not all exceptions can be/are logged.
    #       Some errors may occur before the logger is initialised.
    #       Some standard ones like syntax error and invalid arguments from argparse cannot be logged;
    #       These should not be logged anyway
    if (not os.path.exists('./logs')):
        os.makedirs('./logs')

    fileHandler = logging.FileHandler(filename=f"./logs/{time.strftime('%d%m%Y_%H-%M-%S')}.log") # filename == timestamp.log
    fileHandler.setFormatter(logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s"))
    rootLogger.addHandler(fileHandler)

    sys.excepthook = handle_exception

    # consoleHandler handles piping logging to std.err
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    consoleHandler.setLevel(logging.INFO) # Only store INFO level and above; DEBUG -> INFO -> WARNING -> ERROR -> CRITICAL in ascending severity
    rootLogger.addHandler(consoleHandler)

    # Set Elasticsearch's loggers' levels
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.WARNING)

    # 0b. Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', type=int, help='specify batch size of documents to tag', default=100, choices=range(501))
    parser.add_argument('-o', type=int, help='to set a time (in Unix Epoch Seconds) which documents tagged before this time will be selected for tags overwriting', default=0)
    parser.add_argument('-f', type=int, help='max no. of features of LDA', default=1000)
    parser.add_argument('-i', type=int, help='no. of iterations of LDA', default=1000)
    parser.add_argument('-tw', type=int, help='top k words for a topic to be obtained', default=10, choices=range(31)) # 0-30
    parser.add_argument('-lo', type=int, help='learning offset for LDA', default=50)

    args = parser.parse_args()

    # 0c. Load config variables from .env file; variables loaded prior via tagger.ps1
    #     If variables not found, load defaults
    nodes = []
    if (os.getenv('numNodes') and int(os.getenv('numNodes'))>0): # if no nodes are specified, then load a default of 1 node
        for i in range(1, int(os.getenv('numNodes'))+1):
            nodes.append(os.getenv(f'node{i}'))
    else: # default
        nodes.append('127.0.0.1:9200')
    
    esIndex = os.getenv('esIndex') or 'documents' # load default if not defined in config

    workStartAM = int(os.getenv('workStartHr')) if os.getenv('workStartHr') else 8
    workEndHr = int(os.getenv('workEndHr')) if os.getenv('workEndHr') else 18

    # 1. Program start
    rootLogger.info('Beginning Tagger')
    rootLogger.debug(args)
    check_time(workStartAM, workEndHr, rootLogger) # Quick check if supposed to run

    # 2. Connect to ES Database
    rootLogger.info('Connecting to ES DB...')
    start = time.time()
    es = connectDB(esIndex, nodes, rootLogger)
    end = time.time()
    rootLogger.info(f'Time taken to connect to ES DB: {round(end-start)}s')

    # 3. Load custom stopwords
    rootLogger.info('Loading custom stopwords .txt file...')
    start = time.time()
    stop_words = load_stop_words('./utils/stopwords-master.txt')
    end = time.time()
    rootLogger.info(f'Time taken to load custom stopwords: {round(end-start)}s')

    # 4. Grab documents via _msearch
    while (True):
        check_time(workStartAM, workEndHr, rootLogger) # time-check to know whether to exit during tagger run-time (reached 8 AM of next day)
        
        rootLogger.info('Constructing new Elasticsearch query...')
        start = time.time()
        res = execute_es_query(es, esIndex, args.b, args.o)
        end = time.time()

        numDocsToProcess = sum([len(i['hits']['hits']) for i in res['responses']])
        if (numDocsToProcess < 1):
            # No documents to process
            rootLogger.info('Tagging completed. There are no more documents that require tagging. Tagger will now exit and sleep until next 6PM...')
            sys.exit(99)

        rootLogger.info(f'{numDocsToProcess} document(s) need to be tagged. Time taken to find them: {round(end-start)}s.')

        # 4.5 Process documents
        rootLogger.info(f'Processing {numDocsToProcess} documents...')
        start = time.time()

        # Construct actions array to be passed into bulk helper
        actions = [j for j in get_actions(res['responses'], esIndex, rootLogger, stop_words, args.f, args.i, args.lo, args.tw)]
        end = time.time()
        rootLogger.info(f'Time taken to process {numDocsToProcess} documents: {round(end-start)}s')
        
        # Bulk insertion
        start = time.time()
        bulk(client=es, actions=actions)
        end = time.time()
        rootLogger.info(f'Time taken to perform bulk insertion of tags for {numDocsToProcess} documents: {round(end-start)}s')

        # Refresh documents index after all the updating
        IndicesClient.refresh(es, index=esIndex) 