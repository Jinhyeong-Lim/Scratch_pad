from tensorflow.python.keras import preprocessing
import tf_utils
samples = ['현재날씨는 10분 단위로 갱신되며, 날씨 아이콘은 강수가 있는 경우에만 제공됩니다.',
           '낙뢰 예보는 초단기예보에서만 제공됩니다.',
           '나 좋은 일이 생겼어',
           '아 오늘 진짜 짜증나']

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(samples)

word_index = tokenizer.word_index
print("각 단어의 인덱스: \n", word_index)

sequences = tokenizer.texts_to_sequences(samples)
print(sequences)
