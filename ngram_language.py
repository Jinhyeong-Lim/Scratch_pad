from konlpy.tag import Komoran
from konlpy.tag import Komoran
from collections import defaultdict
import pandas as pd
komoran = Komoran()
doc=[]
komoran = Komoran()

data = pd.read_csv('Test.txt')
doc = data['document']

def get_ngram_counter(docs, min_count=10, n_range=(1,3)):

    def to_ngrams(words, n):
        ngrams = []
        for b in range(0, len(words) - n + 1):
            ngrams.append(tuple(words[b:b+n]))
        return ngrams

    n_begin, n_end = n_range
    ngram_counter = defaultdict(int)
    for doc in docs:
        words = komoran.pos(doc, join=True)
        for n in range(n_begin, n_end + 1):
            for ngram in to_ngrams(words, n):
                ngram_counter[ngram] += 1

    ngram_counter = {
        ngram:count for ngram, count in ngram_counter.items()
        if count >= min_count
    }

    return ngram_counter

ngram_counter = get_ngram_counter(doc)
print(ngram_counter)

for ngram, count in sorted(ngram_counter.items(), key=lambda x:-x[1]):
    if ngram[-1] == '이체/NNG':
        print(ngram, count)