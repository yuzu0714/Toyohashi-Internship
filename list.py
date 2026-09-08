import re
from janome.tokenizer import Tokenizer
from collections import Counter
import pandas as pd

tokenizer = Tokenizer()

def text2words(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        surface = token.surface
        if re.fullmatch(r'[\s\n\t　。、！？…「」『』（）・～：；]+', surface):
            continue
        words.append(surface)
    return words

df = pd.read_csv('./wrime-ver2.tsv', sep='\t', usecols=['Sentence'])

all_words = []
for index, row in df.iterrows():
    words = text2words(row['Sentence'])
    all_words.extend(words)

counter = Counter(all_words)
word_list = counter.most_common(10)
print(word_list)