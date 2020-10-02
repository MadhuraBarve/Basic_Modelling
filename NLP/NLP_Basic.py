import os
import nltk
import nltk.corpus
# nltk.download()
# print(os.listdir(nltk.data.find("corpora")))

from nltk.corpus import brown
from nltk.tokenize import word_tokenize
# print(brown.words())

# print(nltk.corpus.gutenberg.fileids())

## Loading shakespeare-hamlet file
hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
# print(hamlet)

# Get the 1st 500 words from hamlet
# for word in hamlet[:500]:
	# print(word,sep=' ',end= ' ')

## Use of tokenization
hamlet.tockens = word_tokenize(hamlet[:50])
print(hamlet.tockens)