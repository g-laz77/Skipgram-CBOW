import sys
import pickle
import numpy as np
import math
from copy import deepcopy

np.set_printoptions(threshold=np.nan)

#  BUILDING VOCABULARY
vocab = {}

path = sys.argv[1]
fileline = ''
#
# with open(path, 'r') as f:
#     for line in f:
#         for word in line.split():
#             if word not in vocab:
#                 vocab[word] = 1
#             else:
#                 vocab[word] += 1
#
# # print vocab, len(vocab)
# count = 0
# to_delete = []
# for word in vocab:
#     if vocab[word] >= 50:
#         count += 1
#     else:
#         to_delete.append(word)
#
# for item in to_delete:
#     del vocab[item]
#
# print len(vocab)  # {word:freq}
#
# pickle.dump(vocab, open('vocab.txt', 'w'))
#
# vocab = pickle.load(open('vocab.txt', 'r'))
# size = len(vocab)
# print size
# word_vec = {}
# index = 0
# for word in vocab:
#     print word, index
#     vector = np.zeros((1, size), dtype=float)
#     vector[0][index] = 1
#     word_vec[word] = vector
#     index += 1
#
# print 'done'
#
# # for word in word_vec:
# #     print word_vec[word]
# pickle.dump(word_vec, open('word_vec', 'w'))  # word_vec = {word:one-hot vector}

word_vec = pickle.load(open('word_vec', 'r'))
vocabsize = len(word_vec)

with open(path, 'r') as f:
    for line in f:
        fileline += line + ' '

split = fileline.split()
print 'Done with split'

index = 0
final_pairs = []
for word in split:
    if word in word_vec:
        pair = []
        if index == 0:
            if split[index + 1] in word_vec:
                pair.append(word_vec[split[index + 1]])
            if split[index + 2] in word_vec:
                pair.append(word_vec[split[index + 2]])
            if pair:
                pair.append(word_vec[split[index]])
                final_pairs.append(pair)
        elif index == 1:
            if split[index - 1] in word_vec:
                pair.append(word_vec[split[index - 1]])
            if split[index + 1] in word_vec:
                pair.append(word_vec[split[index + 1]])
            if split[index + 2] in word_vec:
                pair.append(word_vec[split[index + 2]])
            if pair:
                pair.append(word_vec[split[index]])
                final_pairs.append(pair)
        elif index == len(split) - 1:
            if split[index - 1] in word_vec:
                pair.append(word_vec[split[index - 1]])
            if split[index - 2] in word_vec:
                pair.append(word_vec[split[index - 2]])
            if pair:
                pair.append(word_vec[split[index]])
                final_pairs.append(pair)
        elif index == len(split) - 2:
            if split[index + 1] in word_vec:
                pair.append(word_vec[split[index + 1]])
            if split[index - 1] in word_vec:
                pair.append(word_vec[split[index - 1]])
            if split[index - 2] in word_vec:
                pair.append(word_vec[split[index - 2]])
            if pair:
                pair.append(word_vec[split[index]])
                final_pairs.append(pair)
        else:
            if split[index + 1] in word_vec:
                pair.append(word_vec[split[index + 1]])
            if split[index - 1] in word_vec:
                pair.append(word_vec[split[index - 1]])
            if split[index - 2] in word_vec:
                pair.append(word_vec[split[index - 2]])
            if split[index + 2] in word_vec:
                pair.append(word_vec[split[index + 2]])
            if pair:
                pair.append(word_vec[split[index]])
                final_pairs.append(pair)
        index += 1
    else:
        index += 1

print 'Done with making context pairs'
wih = np.random.rand(vocabsize, 50)
woh = np.random.rand(50, vocabsize)

eeta = 0.01
# print final_pairs


iterations = 100
for _ in range(iterations):
    for pair in final_pairs:
        context_words = [word for word in pair[:-1]]
        middle_word = pair[-1]
        print context_words
        print middle_word
        old_wih = deepcopy(wih)
        old_woh = deepcopy(woh)
        input_product = np.dot(middle_word, wih)
        winput = []
        # for word in context_words:
        #     word = np.array(word)
        #     product = np.dot(word, wih)
        #     winput.append(product)
        # to_add = np.zeros((1, 50))
        # print to_add
        # for matrix in winput:
        #     to_add = np.add(to_add, matrix)
        # print to_add
        # print len(context_words)
        # to_add = np.true_divide(to_add, len(context_words))
        # baks = deepcopy(to_add)
        # print to_add
        output_product = np.dot(input_product, woh)  # Blue matrix formed
        print output_product
        den_sum = 0
        for i in range(len(output_product[0])):
            den_sum += math.exp(output_product[0][i])
        for i in range(len(output_product[0])):
            output_product[0][i] /= den_sum
        error = []
        for cword in context_words:
            word_error = np.subtract(output_product, cword)
            error.append(word_error)
        error_sum = np.zeros((1, vocabsize))
        for vector in error:
            error_sum = np.add(error_sum, vector)

        dw_output = np.outer(np.dot(np.transpose(old_wih), np.transpose(middle_word)), error_sum)
        woh += -eeta * dw_output
        print woh
        # print 'woh'
        dw_input = np.outer(np.transpose(middle_word), np.dot(old_woh, np.transpose(error_sum)))
        wih += -eeta * dw_input
        print wih
        break