import numpy as np 
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#proses tokenize : memecah kalimat menjadi kata-kata.
def tokenize(sentence):
    """
    #membagi kalimat menjadi array kata / token
    split sentence into array of words/tokens 
    a token can be a word or punctuation character, or number
    #token dapat berupa kata atau karakter tanda baca, atau angka
    """
    #pisahkan kalimat menjadi larik kata/token,token dapat berupa kata atau karakter tanda baca, atau angka.
    
    return nltk.word_tokenize(sentence)
    #Sebuah kalimat atau data dapat dipisah menjadi kata-kata dengan kelas word_tokenize() pada modul NLTK.

#proses stem : menghilangkan imbuhan pada suatu kata.
def stem(word):
    """
    stemming = find the root form of the word ( stemming = mencari imbuhan kata)
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    
    return stemmer.stem(word.lower())

#bag_of_word : proses mengubah data teks menjadi vektor yang dapat dipahami oleh komputer
#mentransformasi teks input dari pengguna menjadi bentuk bilangan binner.
def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    #mengembalikan kelas array kata: 1 untuk setiap kata yang diketahui yang ada dalam kalimat, 0 sebaliknya

    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
