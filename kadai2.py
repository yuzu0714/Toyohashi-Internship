import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import re
from janome.tokenizer import Tokenizer
from collections import Counter


tokenizer = Tokenizer()

KEEP_POS = {'名詞', '動詞', '形容詞'}
EXCLUDE_SUBPOS = {'非自立', '形式名詞', '接尾'}
SYMBOL_PATTERN = re.compile(r'^[^\w\u3040-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F]+$', re.UNICODE)

#1文を受け取って、必要な単語だけのリストを返す関数。
def text2words_filtered(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        pos_info = token.part_of_speech.split(',')
        pos    = pos_info[0]    #品詞を調べる
        subpos = pos_info[1]   #品詞細分類を調べる
        base   = token.base_form    #原型を調べる

        if pos not in KEEP_POS:     #必要な品詞でなければスキップ
            continue
        if subpos in EXCLUDE_SUBPOS:    #必要な品詞細分類でなければスキップ
            continue
        if base == '*' or not base:   #原型が空欄ならスキップ
            continue
        if SYMBOL_PATTERN.match(base):  #記号・絵文字ならスキップ
            continue

        words.append(base)  
    return words

#複数の文のリストを受け取って、頻度順の辞書を返す関数。
def build_vocabulary(counter):
    vocabullary = {}
    for index, (word, _) in enumerate(counter.most_common()):   #_は頻度で値を無視
        vocabullary[word] = index
    return vocabullary

#全文を単語に分割し、頻度順の辞書を作る
def texts2vocab(sentences):
    all_words = []
    for text in sentences:
        all_words.extend(text2words_filtered(text))
    counter = Counter(all_words)
    vocab = {}
    for index, (word, _) in enumerate(counter.most_common()):
        vocab[word] = index
    return vocab

#１文をBoWベクトルに変換する
def text2bow(text, vocab):
    vector = [0] * len(vocab)
    for word in text2words_filtered(text):
        if word in vocab:
            vector[vocab[word]] = 1
    return vector

#複数文をBoWベクトルのリストに変換する
def texts2bows(vocab, sentences):
    return [text2bow(text, vocab) for text in sentences]


target_columns = [
    'Sentence',
    'Writer_Joy'
]

df = pd.read_csv(
    './wrime-ver2.tsv',
    sep='\t',
    nrows=20000,
    usecols=target_columns
)

sentences = []
labels = []
for index, row in df.iterrows():
    sentences.append(row['Sentence'])
    labels.append(1 if row['Writer_Joy'] > 0 else 0)

train_sentences = sentences[:18000]
test_sentences = sentences[18000:]
train_labels = labels[:18000]
test_labels = labels[18000:]

# TODO
vocab = texts2vocab(train_sentences) # 分詞、sort、辞書を作る
train_sentence_bows = texts2bows(vocab, train_sentences) # 18000個のbowリストを得る
test_sentence_bows = texts2bows(vocab, test_sentences)

# TODO: train_sentence_bows, test_sentence_bows, train_labels, test_labelsを
# numpy arrayにする

train_sentence_bows = np.array(texts2bows(vocab, train_sentences), dtype=np.int8)
test_sentence_bows  = np.array(texts2bows(vocab, test_sentences),  dtype=np.int8)
train_labels        = np.array(train_labels)
test_labels         = np.array(test_labels)

# 1. 訓練
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(train_sentence_bows, train_labels)

# 2. 予測
test_preds = clf.predict(test_sentence_bows)

f1_macro = f1_score(test_labels, test_preds, average='macro')
print(f"F1-score (Macro): {f1_macro:.4f}")

print("\n詳細報告:")
print(classification_report(test_labels, test_preds, target_names=["Not Joy (0)", "Joy (1)"]))

