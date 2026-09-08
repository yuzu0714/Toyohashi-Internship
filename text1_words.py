import re
from janome.tokenizer import Tokenizer
import pandas as pd

tokenizer = Tokenizer()  

text = "今日の月も白くて明るい。昨日より雲が少なくてキレイな〜 と立ち止まる帰り道。チャリなし生活も悪くない。"

def text2words(text):
    tokens = tokenizer.tokenize(text)  
    words = []
    for token in tokens:
        surface = token.surface
        if re.fullmatch(r'[\s\n\t　。、！？…「」『』（）・～：；]+', surface):
            continue
        words.append(surface)
    return words

print(text2words(text))