import nltk
sentence = """안녕하세요 하하하 왜 이러세요"""
tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print( tagged[0:6])

entities = nltk.chunk.ne_chunk(tagged)
print(entities)
