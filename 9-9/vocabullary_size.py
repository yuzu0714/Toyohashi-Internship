import re
from janome.tokenizer import Tokenizer
from collections import Counter
import pandas as pd

tokenizer = Tokenizer()

# Remove by part-of-speech (particles, conjunctions, auxiliary verbs, symbols)
REMOVE_POS = {'助詞', '接続詞', '助動詞', '記号'}

# Remove by surface form as a fallback (catches anything POS misses)
REMOVE_PATTERN = re.compile(r'[^\w\u3040-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F]+', re.UNICODE)

def text2words_raw(text):
    tokens = tokenizer.tokenize(text)
    return [token.surface for token in tokens]

def text2words_filtered(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        surface = token.surface
        pos = token.part_of_speech.split(',')[0]

        # Remove by POS
        if pos in REMOVE_POS:
            continue

        # Remove if surface contains only non-Japanese / non-alphanumeric characters
        if REMOVE_PATTERN.fullmatch(surface):
            continue

        words.append(surface)
    return words

df = pd.read_csv('./wrime-ver2.tsv', sep='\t', usecols=['Sentence'], nrows=200)

all_words_raw = []
all_words_filtered = []

for _, row in df.iterrows():
    all_words_raw.extend(text2words_raw(row['Sentence']))
    all_words_filtered.extend(text2words_filtered(row['Sentence']))

counter_raw = Counter(all_words_raw)
counter_filtered = Counter(all_words_filtered)

print("=== Before Filtering ===")
print(f"  Vocabulary size (unique words): {len(counter_raw)}")

print("\n=== After Filtering===")
print(f"  Vocabulary size (unique words): {len(counter_filtered)}")

print("\n=== Difference ===")
print(f"  Vocabulary size reduced by: {len(counter_raw) - len(counter_filtered)}")
print(f"  Total tokens reduced by:    {sum(counter_raw.values()) - sum(counter_filtered.values())}")

print("\n=== Top 10 Most Frequent Words (after filtering) ===")
for word, count in counter_filtered.most_common(10):
    print(f"  {word}: {count}")
