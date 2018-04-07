import sys
import os  # for file separator (in linux & windows)
import math
from utils import *

tagged = sys.argv[1]  # 1 - majority, 2 - bi-gram
fileName = sys.argv[2]  # 1 - majority, 2 - bi-gram
model = sys.argv[3]
smoothing = sys.argv[4]

# file paths
lexical_file_path = 'lexical-model.txt'  # train phase output
structural_file_path = 'train-param-file2.txt'  # train phase output
train_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos2.train'

test_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.test'
gold_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.gold'
classification_output_file_path = 'classifications.tagged'  # decode phase output
evaluate_file_path = 'evaluation.eval'  # evaluate phase output


def get_probability(word_tag_counter, tag_counter):
    return round(math.log(1.0 * word_tag_counter / tag_counter, 5))


# this method implements majority classifier
def convert_to_probabilities_map(word_counters_map, tags_counters_map):
    for (segment, tags_map) in word_counters_map.items():
        for (tag, word_tag_counter) in tags_map.items():
            word_counters_map[segment][tag] = get_probability(word_tag_counter, tags_counters_map[tag])


def print_lexical_data(output_file, words_tags_counters_map):
    for (segment, tags_map) in words_tags_counters_map.items():
        output_file.write(segment)
        if len(tags_map) > 0:
            for (tag, prob) in tags_map.items():
                output_file.write('\t')
                output_file.write(tag + '\t' + str(prob))
        output_file.write('\n')


def lexical_train_phase(words_tags_counters_map, tags_counters_map, lexical_file_output_path):
    convert_to_probabilities_map(words_tags_counters_map, tags_counters_map)
    with open(lexical_file_output_path, "w") as output_file:
            print_lexical_data(output_file, words_tags_counters_map)


def get_gram_key(list, index, gram_size):
    key = ""
    for i in range(gram_size):
        if not end_of_sentence(list[index + i]):
            (seg, tag) = list[index + i]
            key += tag + ":"
    return key[:-1]


def get_next_sentence(start_index, train_list):
    sentece = []
    for index in range(start_index, len(train_list)):
        if end_of_sentence(train_list[index]):
            return sentece
        sentece.append(train_list[index])
    return sentece


def build_n_gram_counters_map(train_file_path, n_grams):
    n_gram_tag_counters = dict()
    train_list = load_key_value_file_as_list(train_file_path)
    # TODO: pad with <s> , </s> before and after each sentence
    i = 0
    while i < len(train_list) - n_grams + 1:
        if end_of_sentence(train_list[i]):
            i += 1
            continue
        else:
            sentence = get_next_sentence(i, train_list)
            i += len(sentence)
            sentence.insert(0, ('<s>', '<s>'))
            sentence.append(('<e>', '<e>'))
            for j in range(len(sentence) - n_grams + 1):
                key = get_gram_key(sentence, j, n_grams)
                if n_gram_tag_counters.has_key(key):
                    n_gram_tag_counters[key] += 1
                else:
                    n_gram_tag_counters[key] = 1
    return n_gram_tag_counters


def write_results_on_structural_file(output_file_path, ngram_counters):
    with open(output_file_path, "w") as output:
            write_number_of_ngrams_section(output, ngram_counters)
            write_ngrams_probability_section(output, ngram_counters)


def update_key_probability(prob_map, gram_counters, gram_size, gram_key):
    if gram_size == 0:
        prob_map[gram_key] = 1.0  # TODO: assign probability
    else:
        # get gram counters map
        ith_gram_counters_map = gram_counters[gram_size]
        prev_ith_gram_counters_map = gram_counters[gram_size - 1]
        # get counters
        key_array = gram_key.split(":")
        ith_gram_counter = ith_gram_counters_map[gram_key]
        prev_ith_gram_counter = prev_ith_gram_counters_map[':'.join(key_array[:-1])]
        # add prob:
        prob_map[gram_key] = 1.0 * ith_gram_counter / prev_ith_gram_counter


def build_probability_map(gram_counters):
    all_prob_map = []
    for index, ith_gram_map in enumerate(gram_counters):
        ith_gram_prob_map = dict()
        all_prob_map.append(ith_gram_prob_map)
        for gram_size, (gram_key, gram_counter) in enumerate(ith_gram_map.items()):
            update_key_probability(ith_gram_prob_map, gram_counters, index, gram_key)
    return all_prob_map


def train_first_order_classifier(train_file_path, lexical_file_path, structural_file_path, order):
    # analyze data
    tags_counters_map = dict()
    words_tags_counters_map = build_segment_tags_map(train_file_path, tags_counters_map)
    # run lexical analysis
    lexical_train_phase(words_tags_counters_map, tags_counters_map, lexical_file_path)
    # create map for each order with holds: for bi-gram: Counter(Ti,Ti-1)..
    ngram_counters = []
    for i in range(1, order + 1):
        ngram_counters.append(build_n_gram_counters_map(train_file_path, i))
    all_prob_map = build_probability_map(ngram_counters)
    write_results_on_structural_file(structural_file_path, all_prob_map)

# def decode(lex_params_path, struc_param_path):


# run all phases one after another
train_first_order_classifier(train_file_path, lexical_file_path, structural_file_path, 3)
