import json
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Reading the file
with open("Sarcasm_Headlines_Dataset.json","r") as f:	
	# datastore = json.load(f)
	datastore = json.loads("[" + 
        f.read().replace("}\n{", "},\n{") + 
    "]")

## Restructuring the data
sentences = []
labels = []
urls = []
for item in datastore:
	sentences.append(item["headline"])
	labels.append(item["is_sarcastic"])
	urls.append(item["article_link"])

## Splitting the data into training and testing
training_size = 20000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

## Parameters used in tokenization
vocab_size = 10000
max_length = 100
padding_type = "post"
trunc_type = "post"
embedding_dim=16
oov_tok = "<OOV>"

## tokenizing the sentences
tokenizer = Tokenizer(oov_token = oov_tok,num_words=vocab_size)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length,padding=padding_type,truncating = trunc_type)

# Need this block to get it to work with TensorFlow 2.x
import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
	tf.keras.layers.GlobalAveragePooling1D(),
	tf.keras.layers.Dense(24,activation="relu"),
	tf.keras.layers.Dense(1,activation="sigmoid")
	])

model.compile(loss="binary_crossentropy",optimizer="adam",metrics = ["acc"])
	
num_epochs = 30
history = model.fit(training_padded,training_labels,epochs=num_epochs,
	validation_data =(testing_padded,testing_labels),verbose=2)

sentence = [
"Granny started to fear spiders in the garden might be real",
"The weather today is bright and sunny"
]

sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences,maxlen =max_length,padding = padding_type,truncating=trunc_type)

print(model.predict(padded))