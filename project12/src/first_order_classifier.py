import sys
import os  # for file separator (in linux & windows)
import math

import numpy as np

from utils import *

tagged = sys.argv[1]  # 1 - majority, 2 - bi-gram
fileName = sys.argv[2]  # 1 - majority, 2 - bi-gram
model = sys.argv[3]
smoothing = sys.argv[4]

# file paths
lexical_file_path = 'lexical-model.txt'  # train phase output
structural_file_path = 'structural-model.txt'  # train phase output
train_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.train'

test_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.test'
gold_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.gold'
class_output_path = 'classifications-bigram.tagged'  # decode phase output
evaluate_file_path = 'evaluation-bigram.eval'  # evaluate phase output


def get_probability(word_tag_counter, tag_counter):
    return math.log(1.0 * word_tag_counter / tag_counter)


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
            sentence.insert(0, (START_TAG, START_TAG))
            sentence.append((END_TAG, END_TAG))
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
        prob_map[gram_key] = math.log((1.0 * ith_gram_counter / prev_ith_gram_counter))


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


def is_bigram_line(line):
    return len(line.split('\t')) == 3


def is_unigram_line(line):
    return len(line.split('\t')) == 2


def load_lex_params(lexical_file_path):
    lex_probs = dict()
    with open(lexical_file_path, 'r') as input:
        lex_lines = input.readlines()
        for line in lex_lines:
                update_struct_map(line, lex_probs)
    return lex_probs


def load_struc_params(structural_file_path):
    struct_probs = dict()
    with open(structural_file_path, 'r') as input:
        lex_lines = input.readlines()
        for line in lex_lines:
            if is_bigram_line(line):
                (tags_key, tags_prob) = decode_key_tags(line)
                struct_probs[tags_key] = tags_prob
    return struct_probs


def load_possible_tags(structural_file_path):
    all_tags = []
    with open(structural_file_path, 'r') as input:
        lex_lines = input.readlines()
        for line in lex_lines:
            if is_unigram_line(line):
                (tags_key, tags_prob) = decode_key_tags(line)
                if tags_key != START_TAG and tags_key != END_TAG:
                    all_tags.append(tags_key)
    return all_tags


# def get_all_possible_tags(structural_file_path):
#     structural_map = load_struc_params(structural_file_path)


def get_trained_model(structural_file_path, lexical_file_path):
    structural_map = load_struc_params(structural_file_path)
    lex_map = load_lex_params(lexical_file_path)
    return structural_map, lex_map


def get_trans_prob(prev_state, state, structural_map, smoothing):
    tags = [prev_state, state]
    return float(structural_map.get(get_strc_key(tags), [smoothing])[0])  # smoothing is here!!


def get_emission_prob(lex_map, word, state, smoothing):
    if lex_map.has_key(word):
        pwt = float(lex_map[word].get(state, smoothing)) # smoothing
    else:
        pwt = 0
    return pwt


def get_log_prob(prev_state, state, word, lex_map, structural_map):
    smoothing = math.log(0.0000001)
    ptt = get_trans_prob(prev_state, state, structural_map, smoothing)
    pwt = get_emission_prob(lex_map, word, state, smoothing)
    return 1.0 * (ptt + pwt)


def get_max(states, viterbi_prob_matrix, state, prev_word_index, word, lex_map, strc_map):
    max_state_log_prob = -sys.maxint - 1
    max_state_tag = ""
    max_state_pointer = 0
    for index_tag, s_tag in enumerate(states):
        tmp = viterbi_prob_matrix[prev_word_index, index_tag] + get_log_prob(s_tag, state, word, lex_map, strc_map)
        if tmp > max_state_log_prob:
            max_state_log_prob = tmp
            max_state_tag = s_tag
            max_state_pointer = index_tag
    return max_state_tag, max_state_log_prob, max_state_pointer


def get_termination_step_max(viterbi_prob_matrix, last_word_index, strc_map, states):
    max_state_log_prob = -sys.maxint - 1
    max_state_tag = ""
    max_state_pointer = 0
    for state_index, last_state_prob in enumerate(viterbi_prob_matrix[last_word_index, :]):
        tmp = viterbi_prob_matrix[last_word_index, state_index] + get_trans_prob(states[state_index], END_TAG, strc_map, smoothing)
        if tmp > max_state_log_prob:
            max_state_log_prob = tmp
            max_state_tag = states[state_index]
            max_state_pointer = state_index
    return max_state_tag, max_state_pointer


def sentence_decoder(sentence, trained_model, states, markers_count):
    structural_map, lex_map = trained_model
    all_states = len(states)  # + 2
    s_words = len(sentence)
    # allocations
    viterbi_prob_matrix = np.zeros((s_words, all_states), np.float64)  # [words, states]
    viterbi_path_matrix = np.zeros((s_words, all_states), np.chararray)
    viterbi_pointer_matrix = np.zeros((s_words, all_states), np.int64)
    best_path = np.zeros(s_words, np.chararray)
    s0 = START_TAG
    # initialize values
    for index, state in enumerate(states):
        viterbi_prob_matrix[0, index] = get_log_prob(s0, state, sentence[markers_count / 2].strip(), lex_map, structural_map)
        viterbi_path_matrix[0, index] = s0
        viterbi_pointer_matrix[0, index] = -1  # not needed
    if len(sentence) > 1:
        for i_word, word in enumerate(sentence[1:], start=1):
            for i_state, state in enumerate(states):
                max_tag_str, max_tag_prob, max_tag_pointer = get_max(states, viterbi_prob_matrix, state, i_word - 1, word.strip(), lex_map, structural_map)
                viterbi_prob_matrix[i_word, i_state] = max_tag_prob
                viterbi_path_matrix[i_word, i_state] = max_tag_str
                viterbi_pointer_matrix[i_word, i_state] = max_tag_pointer

    last_tag_str, next_state_pointer = get_termination_step_max(viterbi_prob_matrix, i_word, structural_map, states)
    # back trace
    best_path[s_words - 1] = last_tag_str
    for t in range(s_words - 1, 0, -1):  # states of (last-1)th to 0th time step
        best_path[t - 1] = viterbi_path_matrix[t, next_state_pointer]
        next_state_pointer = viterbi_pointer_matrix[t, next_state_pointer]
    return best_path


def decode(test_file, trained_model, possible_tags, classification_path):
    n_grams = 2
    markers_count = ((n_grams - 1) * 2)
    with open(classification_path, "w") as classification_file:
        # load test file
        with open(test_file, "r") as test_data:
            test_lines = test_data.readlines()
        # all_tags_with_markers_size = len(possible_tags) - markers_count
        i = 0
        while i < len(test_lines) - n_grams + 1:
            if end_of_sentence(test_lines[i]):
                classification_file.write(test_lines[i])
                i += 1
                continue
            else:
                sentence = get_next_sentence(i, test_lines)
                i += len(sentence)
                tagged_sentence = sentence_decoder(sentence, trained_model, possible_tags, markers_count)
                append_sentence_to_classification_file(classification_file, sentence, tagged_sentence)
    print ""


# run all phases one after another
#train_first_order_classifier(train_file_path, lexical_file_path, structural_file_path, 2)
decode(test_file_path, get_trained_model(structural_file_path, lexical_file_path), load_possible_tags(structural_file_path), class_output_path)
evaluate(class_output_path, gold_file_path, evaluate_file_path, test_file_path, 2, smoothing)