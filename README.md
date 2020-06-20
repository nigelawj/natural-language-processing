# Topic Modelling
Starter codes for Topic Modelling 

# lda
Latent Dirichlet Allocation (LDA) with `gensim`, to be paired with Topic Labelling via doc2vec models
- NOTE: newsgroup.json must be extracted from newsgroup.7z

# lda-nmf
Latent Dirichlet Allocation vs Non-negative Matrix Factorisation in `Scikit-Learn`

# stopwwords
Utility module to concatenate multiple stopwords .txt files into one
- Twitter-centric stopwords

# tagger
Codes to setup a daemon script that performs tagging of documents where 10 words (tags) of a document are obtained via LDA, and inserted into an Elasticsearch database.
- By default, the script checks if is allowed to run, and can be passed in a time parameter such that only certain documents are re-tagged to reduce redundant tagging operations
- The daemon script must be configured, such as connecting to the database, and durations when script is allowed to run
- Parameters like number of tags and hyperparameters like learning rate can be tweaked as required
- Code will require setting up, it will not work out-of-the-box