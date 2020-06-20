# Gets master stopword list from all stopword
# .txt files in stopword-chunks folder
import glob
import re

stop_words = set()

files = glob.glob('./stopword-chunks/*.txt')
for name in files:
    with open(name, 'r') as f:
        text = f.read()
        words = re.split(r'[;,\s\n]\s*', text)
        stop_words.update(words)

with open('./stopwords-master.txt', 'w') as f:
    for word in stop_words:
        f.write(f'{word}\n')