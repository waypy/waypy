import os
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import pickle
import random
import json

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from waypy.settings.base import BASE_DIR, DEBUG

factory = StemmerFactory()
stemmer = factory.create_stemmer()

with open(os.path.join(BASE_DIR, 'modules/classifier/training/common.json')) as json_data:
    intents = json.load(json_data)

words = list()
classes = list()
documents = list()
ignore_words = ['?']

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

# create our training data
training = list()
output = list()
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = list()
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir=os.path.join(BASE_DIR, 'modules/classifier/tflearn_logs'))
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=DEBUG)
model.save(os.path.join(BASE_DIR, 'modules/classifier/training/data/{}'.format('model.tflearn')))

# load tf model
model.load(os.path.join(BASE_DIR, 'modules/classifier/training/data/{}'.format('model.tflearn')))


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, text, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    part = [0] * len(text)
    for s in sentence_words:
        for i, word in enumerate(text):
            if word == s:
                part[i] = 1
                if show_details:
                    print("found in part: %s" % w)

    return np.array(part)


def main(text, past=None):
    p = bow(text, words, DEBUG)
    responses = next((item['responses'] for item in intents['intents']
                      if item['tag'] == classes[int(np.argmax(model.predict([p])))]), False)
    i = len(responses)
    if DEBUG:
        print(p)
        print(classes)
        print(model.predict([p]))
        print(np.argmax(model.predict([p])))
        print(classes[int(np.argmax(model.predict([p])))])

    result = past

    try:
        result = responses[random.randint(0, i)]
    except IndexError:
        pass

    return result


# save all of our data structures
pickle.dump({
    'words': words,
    'classes': classes,
    'train_x': train_x,
    'train_y': train_y
}, open(os.path.join(BASE_DIR, 'modules/classifier/training/data/{}'.format('training_data')), 'wb'))
