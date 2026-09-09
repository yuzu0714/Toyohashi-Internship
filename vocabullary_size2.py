import re
from janome.tokenizer import Tokenizer
from collections import Counter
import pandas as pd

tokenizer = Tokenizer()

KEEP_POS = {'名詞', '動詞', '形容詞'}
EXCLUDE_SUBPOS = {'非自立', '形式名詞', '接尾'}
SYMBOL_PATTERN = re.compile(r'^[^\w\u3040-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F]+$', re.UNICODE)


def text2words_raw(text):
    tokens = tokenizer.tokenize(text)
    return [token.surface for token in tokens]

def text2words_filtered(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        pos_info = token.part_of_speech.split(',')
        pos      = pos_info[0]
        subpos   = pos_info[1]
        base     = token.base_form

        if pos not in KEEP_POS:
            continue
        if subpos in EXCLUDE_SUBPOS:
            continue
        if base == '*' or not base:
            continue

        # 記号・絵文字を除外
        if SYMBOL_PATTERN.match(base):
            continue

        words.append(base)
    return words

df = pd.read_csv('./wrime-ver2.tsv', sep='\t', usecols=['Sentence'], nrows=200)

all_words_raw      = []
all_words_filtered = []

for _, row in df.iterrows():
    all_words_raw.extend(text2words_raw(row['Sentence']))
    all_words_filtered.extend(text2words_filtered(row['Sentence']))

counter_raw      = Counter(all_words_raw)
counter_filtered = Counter(all_words_filtered)

print("=== Before Filtering ===")
print(f"  Vocabulary size (unique words): {len(counter_raw)}")
print(f"  Total tokens:                  {sum(counter_raw.values())}")

print("\n=== After Filtering (noun / verb / adjective, base form) ===")
print(f"  Vocabulary size (unique words): {len(counter_filtered)}")
print(f"  Total tokens:                  {sum(counter_filtered.values())}")

print("\n=== Difference ===")
print(f"  Vocabulary size reduced by: {len(counter_raw) - len(counter_filtered)}")
print(f"  Total tokens reduced by:    {sum(counter_raw.values()) - sum(counter_filtered.values())}")

print("\n=== Top 10 Most Frequent Words (after filtering) ===")
for word, count in counter_filtered.most_common(10):
    print(f"  {word}: {count}")