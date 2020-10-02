import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
'I love my dog',
'I love my cat',
'you love My dog!',
'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100,oov_token= "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

## to make the length of all the setences equal
padded = pad_sequences(sequences)# to get padding before actaul
padded = pad_sequences(sequences,padding='post') # to get padding after actaul
padded = pad_sequences(sequences,padding='post',truncating = "post",maxlen=5) # to get padding after actaul and the sentence length as the max length

print(word_index)
print(sequences)
print(padded)

test_data = [
"I really love my dog",
"my dog loves my manatee"
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)