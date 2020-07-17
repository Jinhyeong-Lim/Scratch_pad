bards_words = ["The fool doth think he is wise,", "but the wise man knows himself to be a fool"]

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(bards_words)


print(format(len(vect.vocabulary_)))

print(format(vect.vocabulary_))

bag_of_words = vect.transform(bards_words)
print(bag_of_words.toarray())