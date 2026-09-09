import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import re
from janome.tokenizer import Tokenizer
from collections import Counter
from sklearn.metrics import confusion_matrix

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

def texts2vocab(sentences):
    all_words = []
    for text in sentences:
        all_words.extend(text2words_filtered(text))
    counter = Counter(all_words)
    vocab = {}
    for index, (word, _) in enumerate(counter.most_common()):
        vocab[word] = index
    return vocab

def text2bow(text, vocab):
    vector = [0] * len(vocab)
    for word in text2words_filtered(text):
        if word in vocab:
            vector[vocab[word]] = 1
    return vector

def texts2bows(vocab, sentences):
    return [text2bow(text, vocab) for text in sentences]


# データ読み込み（200行）
df_vocab = pd.read_csv(
    './wrime-ver2.tsv',
    sep='\t',
    nrows=10000,
    usecols=['Sentence']
)

df = pd.read_csv(
    './wrime-ver2.tsv',
    sep='\t',
    nrows=200,
    usecols=['Sentence']
)

# 訓練データ（0~99行目）
train_sentences = df['Sentence'].iloc[:100].tolist()
train_labels = [0,1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0]

# テストデータ（100~199行目）
test_sentences  = df['Sentence'].iloc[100:200].tolist()
test_labels = [0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0]
vocab = texts2vocab(df_vocab['Sentence'].tolist())
train_sentence_bows = np.array(texts2bows(vocab, train_sentences), dtype=np.int8)
test_sentence_bows  = np.array(texts2bows(vocab, test_sentences),  dtype=np.int8)
train_labels        = np.array(train_labels)
test_labels         = np.array(test_labels)

# 訓練
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(train_sentence_bows, train_labels)

# 予測・評価
test_preds = clf.predict(test_sentence_bows)

f1_macro = f1_score(test_labels, test_preds, average='macro')
print(f"F1-score (Macro): {f1_macro:.4f}")

print("\n詳細報告:")
print(classification_report(test_labels, test_preds, target_names=["Not Satisfied (0)", "Satisfied (1)"]))

print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, test_preds)
print(cm)

#予測・評価を表示する
print(test_preds)
