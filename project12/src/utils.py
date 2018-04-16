import re  # for splitting by tabs
from itertools import izip
import math

import sys
# constants
MISSING = object()
START_TAG = "<s>"
END_TAG = "<e>"
MAX_NEGATIVE = -sys.maxint - 1
NNP_STATE = "NNP"
SMOOTHING_FACTOR = math.log(0.0000001)


def is_line_legal(line):
    return len(line.strip()) > 0 and not line.startswith("#") and len(re.split(r'\t+', line)) == 2


# for gold & train files
def load_key_value_file_as_list(file_path):
    output_list = []
    with open(file_path, "r") as file_object:
        for line in file_object.readlines():
            if is_line_legal(line):
                tokens = re.split(r'\t+', line)
                output_list.append((tokens[0].strip(), tokens[1].strip()))
            else:
                output_list.append(line)  # end of sentence mark
    return output_list


def end_of_sentence(list_item):
    return not isinstance(list_item, tuple) and list_item.strip() == ''


def is_comment_line(line):
    return line.startswith("#")


def print_title_to_file(file_to_print, title):
    file_to_print.write("#-----------------------------------------------------------\n")
    file_to_print.write("# " + title + "\n")
    file_to_print.write("#-----------------------------------------------------------\n")


def print_eval_details(file_to_print, model, smoothing, test_file, gold_file):
    file_to_print.write("#\n")
    file_to_print.write("# Model: " + str(model) + "\n")
    file_to_print.write("# Smoothing: " + str(smoothing) + "\n")
    file_to_print.write("# Test File: " + test_file + "\n")
    file_to_print.write("# Gold File: " + gold_file + "\n")
    file_to_print.write("#\n")


def print_sentence_data(file_to_print, data_list):
    for index, (sen_index, seg_acc, sen_acc) in enumerate(data_list):
        file_to_print.write(str(sen_index) + " " + str(seg_acc) + " " + str(sen_acc) + "\n")
    file_to_print.write("#-----------------------------------------------------------\n")


def print_macro_avg(file_to_print, seg_acc, sen_acc):
    file_to_print.write("macro-avg " + str(seg_acc) + " " + str(sen_acc) + "\n")


def write_number_of_ngrams_section(output_file, ngrams_data):
    output_file.write("\data\\ \n")
    for index, elem in enumerate(ngrams_data):
        output_file.write("ngram " + str(index + 1) + " = " + str(len(elem)) + "\n")
    output_file.write("\n")


def split_tags(elem):
    return elem.split(":")


def decode_key_tags(lex_line):
    cells = lex_line.strip().split("\t")
    return get_strc_key(cells[1:]).strip(), cells[:1]


def get_strc_key(tags):
    return ', '.join(tags)


def update_struct_map(lex_line, segment_map):
    cells = lex_line.split("\t")
    pair_iter = iter(cells[1:])
    for tag, prob in izip(pair_iter, pair_iter):
        if not segment_map.has_key(cells[0]):
            segment_map[cells[0]] = dict()
        segment_map[cells[0]][tag] = prob.strip()
    return segment_map


def write_ngrams_probability_section(output_file, ngrams_data):
    for index, elem in enumerate(ngrams_data):
        output_file.write("\\" + str(index + 1) + "-grams\\ \n")
        for gram_index, (gram_key, gram_prob) in enumerate(elem.items()):
            output_file.write(str(gram_prob) + "\t" + gram_key.replace(":", "\t") + "\n")
        output_file.write("\n")
    output_file.write("\n")


# takes a map and update key counter (initialize OR increase)
def update_key_counter(map_to_use, key):
    # update tags dictionary
    if map_to_use.has_key(key):
        map_to_use[key] += 1
    else:
        map_to_use[key] = 1


def update_words_tag_model(words_tag_dict, word_segment, word_tag):
    # update words tag dictionary
    if words_tag_dict.has_key(word_segment):
        if words_tag_dict[word_segment].has_key(word_tag):
            words_tag_dict[word_segment][word_tag] += 1
        else:
            words_tag_dict[word_segment][word_tag] = 1  # init tag counter
    else:
        words_tag_dict[word_segment] = dict()
        words_tag_dict[word_segment][word_tag] = 1  # init tag counter


# this method build temporary model which hold for
# each segment type a map of all POS tags with counter for
# each one of them: SEG -> {t1:3, t2:4, t7: 15}
def build_segment_tags_map(train_file_path, tags_map=MISSING):
    print "train on file: " + train_file_path
    uni_count = 0
    words_tag_model_tmp_data = dict()
    # read data from train file
    with open(train_file_path, "r") as train_file:
        train_lines = train_file.readlines()
    # build word-tag model
    for line in train_lines:
        if is_comment_line(line):
            continue
        uni_count += 1
        if is_line_legal(line):
            tokens = re.split(r'\t+', line)
            segment_tag = tokens[1].strip('\n')
            segment_word = tokens[0].strip('\n')
            update_words_tag_model(words_tag_model_tmp_data, segment_word, segment_tag)
            if not tags_map == MISSING:
                update_key_counter(tags_map, segment_tag)
    return words_tag_model_tmp_data


def append_sentence_to_classification_file(classification_file, sentence_words, sentence_class):
        for i, word in enumerate(sentence_words):
            classification_file.write(str(sentence_words[i]).strip() + '\t' + str(sentence_class[i]).strip() + '\n')


def word_accuracy_for_sentence(sentence_matches, sentence_length):
    return 1.0 * sentence_matches / sentence_length


def sentence_accuracy_for_sentence(sentence_matches, sentence_length):
    if word_accuracy_for_sentence(sentence_matches, sentence_length) >= 1.0:
        return 1
    return 0


# this method takes a sentence details and update output list with new values
def update_accuracy_list(sentences_accuracy_list, sentence_index, last_sentence_matches, last_sentence_length):
    word_acc = word_accuracy_for_sentence(last_sentence_matches, last_sentence_length)
    sent_acc = sentence_accuracy_for_sentence(last_sentence_matches, last_sentence_length)
    sentences_accuracy_list.append((sentence_index, word_acc, sent_acc))


def sentence_accuracy_for_test_corpus(sentences_accuracy_list):
    perfect_sentences = len(filter(lambda ((sentence_index, word_acc, sent_acc)): sent_acc == 1, sentences_accuracy_list))
    return 1.0 * perfect_sentences / len(sentences_accuracy_list)


def add_to_conf_matrix(conf_matrix, real_tag, prediction):
    if conf_matrix != MISSING:
        if conf_matrix.has_key(real_tag):
            conf_matrix[real_tag][prediction] += 1
        else:
            conf_matrix[real_tag] = dict()
            conf_matrix[real_tag][prediction] = 1


def evaluate(classification_output_path, gold_path, evaluate_file_path, test_file_path, model, smoothing, conf_matrix=MISSING):
    sentences_accuracy_list = []
    sentence_index = 0
    last_sentence_length = 0
    last_sentence_matches = 0
    matches = 0
    errors = 0
    all_data = 0
    classifications_file_items = load_key_value_file_as_list(classification_output_path)
    gold_file_items = load_key_value_file_as_list(gold_path)
    for index, elem in enumerate(gold_file_items):
        classification_item = gold_file_items[index]
        if end_of_sentence(classification_item) and last_sentence_length > 0:
            sentence_index += 1
            update_accuracy_list(sentences_accuracy_list, sentence_index, last_sentence_matches, last_sentence_length)
            last_sentence_length = 0
            last_sentence_matches = 0
        else:  # segment with tag, compare tags
            prediction_item = classifications_file_items[index]
            (class_segment, prediction) = prediction_item
            (gold_segment, real_tag) = classification_item
            if prediction == real_tag:
                matches += 1
                last_sentence_matches += 1
            else:
                errors += 1
            add_to_conf_matrix(conf_matrix, real_tag, prediction)
            all_data += 1
            last_sentence_length += 1
    # last sentence can finish without EOL character
    if last_sentence_length > 0:
        update_accuracy_list(sentences_accuracy_list, sentence_index, last_sentence_matches, last_sentence_length)
    # write results to file
    with open(evaluate_file_path, "w") as eval_file:
        print_title_to_file(eval_file, "Part-of-speech Tagging Evaluation")
        print_eval_details(eval_file, model, smoothing, test_file_path, gold_path)
        print_title_to_file(eval_file, "sent-num word-accuracy sent-accuracy")
        print_sentence_data(eval_file, sentences_accuracy_list)
        print_macro_avg(eval_file, 1.0 * matches / all_data, sentence_accuracy_for_test_corpus(sentences_accuracy_list))
    return conf_matrix
