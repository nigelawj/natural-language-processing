'''
This file consists of tests related to the pre-processing and performing of LDA on a sample text
'''
print('\n\nSTART OF TEST\n\n')
# Imports

# LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import re

# Load stopwords
print('Loading Stopwords...')
stop_words = set()
with open('../stopwords-master.txt', 'r') as f:
    text = f.read()
    stop_words.update(re.split(r'[;,\s\n]\s*', text))
stop_words = set(stop_words)

# Pre-processing
doc = '1 CEO DSO’S SPEECH AT LAUNCH OF THE SATELLITE TECHNOLOGY AND RESEARCH CENTRE ON 25 JANUARY 2018 Professor Ho Teck Hua, Senior Deputy President and Provost, National University of Singapore Mr. Beh Kian Teik, Assistant Managing Director and Chief Digital Officer, EDB Mr. Tang Kum Chuen, Deputy President (Corporate Development) and President (Satellite Systems), ST Electronics Mr. Jarry Herve, Chief Technical Officer, Thales Solutions Asia Ladies and Gentlemen, A very good morning. I am delighted to be here today for the launch of the Satellite Technology And Research Centre (or STAR Centre in short). DSO’s Journey in Space As a national R&D organisation, DSO invests in nascent technologies that could potentially become mainstream some 15 to 20 years later. Back in the late 1990’s, we identified satellite technology as an area of enormous promise. Apart from the value it could bring to Singapore, satellites are an exciting frontier, and the pinnacle of engineering. It is an area that excites and inspires. 2 In 2001, DSO partnered the Nanyang Technological University (or NTU) to form Centre for Research in Satellite Technologies (or CREST), subsequently renamed Satellite Research Center (or SaRC). Together, we set our sights on developing a micro-satellite. This first foray into space had (more than) its fair share of challenges. But we persevered and prevailed. In 2011, the X-SAT was successfully launched. Singapore had entered space – with its own indigenous development. The X-SAT proved to be a valuable tool for remote sensing, particularly for environmental monitoring purposes. It created excitement among technologists in the local ecosystem. Our journey in space reached new heights with the successful launch in 2015 of six locally built satellites, including micro-satellite payloads from NTU and NUS; and the first commercial electro-optical mini-satellite from ST Electronics – theTeLEOS-1. TeLEOS-1 is a Near Equatorial Orbit Earth Observation satellite. It provides insightful, relevant, and timely geo-spatial solutions for Maritime Security and Safety, Humanitarian Aid and Disaster Relief, Forestation and Agriculture, and Infrastructure Construction Progress, to name a few. TeLEOS-1 is the result of collaboration among our key technology players via the Joint Venture between DSO, NTU, and ST Electronics (Satcom & Sensor Systems). ST Electronics led 3 the design, development, system integration, and testing of the satellite while the joint team from DSO and CREST at NTU provided the systems engineering expertise to ensure transition from research to industrialisation. Additionally, CRISP at NUS assisted the project in image processing. Local satellite development benefitted from EDB’s support. EDB sees potential in the satellite sector. It believed in the local ecosystem and provided funding for some of the payloads in the 2015 six-in-one satellite launch. The success of TeLEOS-1 underscored the technical capability of the local ecosystem. It gave us the confidence to think where this could lead us. Located near the equator, Singapore is ideally positioned to develop and operate low inclination angle satellites to provide high temporal remote sensing and geospatial solutions for environmental monitoring, maritime situation awareness, and disaster management. We envision that if we could leverage our natural geographic advantage and harness emergent nanosatellite constellation technology, we could develop low-cost satellites in numbers to yield re-visit rates that are ideal for near real-time applications. EDB and ST Electronics (Satcom & Sensor Systems) share the same vision. Apart from economic value, the idea fits well into the overall vision for developing the satellite sector in Singapore. 4 5 STAR in Space The focal point of research to support the nanosatellite constellation idea is STAR Centre. STAR was formed under the ambit of an MOU between NUS and DSO back in February 2017 with the mission to be a leader in advanced distributed small satellite systems. It will develop cutting-edge capabilities in distributed satellite systems. STAR researchers worked in a laboratory at the NUS Faculty of Engineering for several months and moved to their permanent premises at SWIFT in November. I am pleased that STAR has a place to call home. I am optimistic that it would conduct outstanding research and fulfil the role. STAR is the latest in the longstanding collaboration between NUS and DSO. This collaboration is mutually beneficial, enabling DSO to tap the scientific talent in NUS to develop advanced technologies in areas relevant to defence. It has also expanded the technological base in Singapore; and invigorated interest in Science, Technology, Engineering and Mathematics (or STEM) among new cohorts of young Singaporeans. STAR will no doubt further reinforce this cooperation. 6 Conclusion STAR was born through the unwavering support and dedication of the team at NUS and my fellow colleagues. But it would not have been possible too without the belief and encouragement of EDB. I would like to take this opportunity to thank EDB for supporting us with the necessary resources to develop our team and fund our research in this area and for helping us construct valuable linkages with other research industries. With that, I wish STAR success in achieving its vision. Thank you. Word count: 790'

# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(doc)
# convert to lower case
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
stripped_text = [word for word in stripped if word.isalpha()]
# filter out stop words
stripped_text = [w for w in stripped_text if not w in stop_words]

doc = ' '.join(stripped_text)
print('\n\nSTRIPPED TEXT:')
print(doc)

no_features = 1000

# LDA
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform([doc]) # fit on 1 document
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 1

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=1000, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10

# Populate terms and columns for doc2vec input
lda_columns = []
for i in range(0, no_top_words):
    lda_columns.append('term{}'.format(i))

lda_terms = []
for topic_idx, topic in enumerate(lda.components_):
    lda_terms.append([tf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])

lda_df = pd.DataFrame(data=lda_terms, columns=lda_columns)
lda_df['topic_id'] = [i for i in range(0, no_topics)]
lda_df.set_index('topic_id', inplace=True)
print('\nLDA:\n')
print(lda_df.head())

# NMF
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform([doc]) # fit on 1 document
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

# NMF
nmf_columns = []
for i in range(0, no_top_words):
    nmf_columns.append('term{}'.format(i))

nmf_terms = []
for topic_idx, topic in enumerate(nmf.components_):
    nmf_terms.append([tfidf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])

nmf_df = pd.DataFrame(data=nmf_terms, columns=nmf_columns)
nmf_df['topic_id'] = [i for i in range(0, no_topics)]
nmf_df.set_index('topic_id', inplace=True)
print('\nNMF:\n')
print(nmf_df.head())

print('\n\n\nEND OF TEST')


