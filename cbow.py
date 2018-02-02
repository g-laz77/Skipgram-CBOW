import urllib.request
import collections
import math
import os
import random
import zipfile
import datetime as dt
from itertools import compress
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import *
from tensorflow.python.framework import *
# from backprop import *

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def collect_data(vocabulary_size=10000):
    filename = 'full_text_sentences_new.zip'
    vocabulary = read_data(filename)
    # print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, num_skips*2), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span) # used for collecting data[data_index] in the sliding window
    # collect the first window of words
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # move the sliding window  
    for i in range(batch_size):
        mask = [1] * span
        mask[skip_window] = 0 
        batch[i, :] = list(compress(buffer, mask)) # all surrounding words
        labels[i, 0] = buffer[skip_window] # the word at the center 
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch,labels

vocabulary_size = 10000
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocabulary_size)
batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
learning_rate=0.1
graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size, num_skips*2])  
  train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  print(train_dataset)
  print(train_context)
  print(valid_dataset)

  # Look up embeddings for inputs.
  embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  print(embeddings)
  embed = tf.zeros([batch_size, embedding_size])
  for j in range(num_skips):
    embed += tf.nn.embedding_lookup(embeddings, train_dataset[:, j])
  print(embed)

  # Construct the variables for the softmax
  weights = tf.Variable(
      tf.truncated_normal([embedding_size, vocabulary_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
  biases = tf.Variable(tf.zeros([vocabulary_size]))
  hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(embed))) + biases

  # convert train_context to a one-hot format
  train_one_hot = tf.one_hot(train_context, vocabulary_size)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))

  grad_W, grad_b = tf.gradients(xs=[weights, biases], ys=cross_entropy)
  weights = weights.assign(weights - learning_rate * grad_W)
  biases = biases.assign(biases - learning_rate * grad_b)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()


def run(graph, num_steps):
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_context = generate_batch(data,batch_size, num_skips, skip_window)
        feed_dict = {train_dataset: batch_inputs, train_context: batch_context}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _ , _, loss_val = session.run([weights,biases, cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 1000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
      final_embeddings = normalized_embeddings.eval()

num_steps = 4000
softmax_start_time = dt.datetime.now()
run(graph, num_steps=num_steps)
softmax_end_time = dt.datetime.now()
print("Softmax method took {} seconds to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))