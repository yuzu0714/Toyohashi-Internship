import re
from janome.tokenizer import Tokenizer
from collections import Counter
import pandas as pd

tokenizer = Tokenizer()

KEEP_POS = {'名詞', '動詞', '形容詞'}
EXCLUDE_SUBPOS = {'非自立', '形式名詞', '接尾'}
SYMBOL_PATTERN = re.compile(r'^[^\w\u3040-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F]+$', re.UNICODE)


def text2words_filtered(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        pos_info = token.part_of_speech.split(',')
        pos    = pos_info[0]
        subpos = pos_info[1]
        base   = token.base_form

        if pos not in KEEP_POS:
            continue
        if subpos in EXCLUDE_SUBPOS:
            continue
        if base == '*' or not base:
            continue
        if SYMBOL_PATTERN.match(base):
            continue

        words.append(base)
    return words


def build_vocabulary(counter):
    vocabullary = {}
    for index, (word, _) in enumerate(counter.most_common()):
        vocabullary[word] = index
    return vocabullary


def text2bow(text):
    vector = [0] * len(vocabullary)
    for word in text2words_filtered(text):
        if word in vocabullary:
            vector[vocabullary[word]] = 1
    return vector


df = pd.read_csv('./wrime-ver2.tsv', sep='\t', usecols=['Sentence'], nrows=200)

all_words_filtered = []
for _, row in df.iterrows():
    all_words_filtered.extend(text2words_filtered(row['Sentence']))

counter_filtered = Counter(all_words_filtered)
vocabullary = build_vocabulary(counter_filtered)

# 行番号を指定してBoWに変換
row_index = 0   # ← ここを変えると別の行を読める
text = df['Sentence'].iloc[row_index]
bow  = text2bow(text)

print(f"Row {row_index}: {text}")
print(f"BoW vector: {bow}")