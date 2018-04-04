import sys
import re  # for splitting by tabs
import os  # for file separator (in linux & windows)
import json  # for writing the parameter file
from utils import *

tagged = sys.argv[1]  # 1 - majority, 2 - bi-gram
fileName = sys.argv[2]  # 1 - majority, 2 - bi-gram
model = sys.argv[3]
smoothing = sys.argv[4]

# file paths
param_file_path = 'train-param-file.txt'  # train phase output
train_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.train'
test_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.test'
gold_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.gold'
classification_output_file_path = 'classifications.tagged'  # decode phase output
evaluate_file_path = 'evaluation.eval'  # evaluate phase output


def update_words_tag_model(words_tag_dict, word_segment, word_tag):
    # calc average tags per word
    if words_tag_dict.has_key(word_segment):
        if words_tag_dict[word_segment].has_key(word_tag):
            words_tag_dict[word_segment][word_tag] += 1
        else:
            words_tag_dict[word_segment][word_tag] = 1  # init tag counter
    else:
        words_tag_dict[word_segment] = dict()
        words_tag_dict[word_segment][word_tag] = 1  # init tag counter


# this method implements majority classifier
def get_majority_tag(tags_map):
    max_tag_counter = 0
    max_tag_key = ""
    for (tag, tag_counter) in tags_map.items():
        if tag_counter > max_tag_counter:
            max_tag_key = tag
            max_tag_counter = tag_counter
    return max_tag_key


def train_phase(train_file_path, param_file_path_output):
    uni_count = 0
    words_tag_model_tmp_data = dict()
    words_tag_model_output = dict()
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
            update_words_tag_model(words_tag_model_tmp_data, tokens[0].strip('\n'), tokens[1].strip('\n'))
    # filter only majority tag for each word segment
    for (segment, tags_map) in words_tag_model_tmp_data.items():
        words_tag_model_output[segment] = get_majority_tag(tags_map)
    # writing the output as parameter file
    with open(param_file_path_output, "w") as model_output:
        model_output.write(json.dumps(words_tag_model_output))


def classify_phase(param_path, test_path, classification_path):
    classifications = []
    # get trained model from parameters file
    with open(param_path, "r") as model_output:
        trained_model = json.loads(model_output.read())
    # get test file data
    with open(test_path, "r") as test_data:
        test_lines = test_data.readlines()
    for line in test_lines:
        if is_comment_line(line):
            continue
        if not line.strip('\n') == "":
            segment = line.strip('\n')
            classifications.append((segment, trained_model.get(segment, "NNP")))
        else:
            classifications.append(line)  # for saving the spaces between sentences in classification file
    # write classifications to file
    with open(classification_path, "w") as classification_file:
        for item in classifications:
            if is_comment_line(line):
                continue
            if not end_of_sentence(item):
                (segment, classification) = item
                classification_file.write(segment + '\t' + classification + '\n')
            else:
                classification_file.write(item)


def sentence_accuracy_for_test_corpus(sentences_accuracy_list):
    perfect_sentences = len(filter(lambda ((sentence_index, word_acc, sent_acc)): sent_acc == 1, sentences_accuracy_list))
    return 1.0 * perfect_sentences / len(sentences_accuracy_list)


def word_accuracy_for_sentence(sentence_matches, sentence_length):
    return 1.0 * sentence_matches / sentence_length


def sentence_accuracy_for_sentence(sentence_matches, sentence_length):
    if word_accuracy_for_sentence(sentence_matches, sentence_length) >= 1.0:
        return 1
    return 0


def print_evaluate_file(sentences_accuracy_list, matches, all_data):
    print "evaluations for basic analyzer:"
    print "     matches " + str(matches) + " all data " + str(all_data)
    print "     sentence accuracy for test corpus " + str(sentence_accuracy_for_test_corpus(sentences_accuracy_list))


def evaluate(classification_output_path, gold_path, evaluate_file_path):
    sentences_accuracy_list = []
    sentence_index = 1
    last_sentence_length = 0
    last_sentence_matches = 0
    matches = 0
    errors = 0
    all_data = 0
    classifications_file_items = load_key_value_file_as_list(classification_output_path)
    gold_file_items = load_key_value_file_as_list(gold_path)
    for index, elem in enumerate(gold_file_items):
        prediction_item = classifications_file_items[index]
        classification_item = gold_file_items[index]
        if end_of_sentence(classification_item):
            word_acc = word_accuracy_for_sentence(last_sentence_matches, last_sentence_length)
            sent_acc = sentence_accuracy_for_sentence(last_sentence_matches, last_sentence_length)
            sentences_accuracy_list.append((sentence_index, word_acc, sent_acc))
            if last_sentence_length > 0:  # if we had words in sentence and it's not just multiple EOL characters
                sentence_index += 1
            last_sentence_length = 0
            last_sentence_matches = 0
        else:  # segment with tag, compare tags
            (class_segment, prediction) = prediction_item
            (gold_segment, real_tag) = classification_item
            if prediction == real_tag:
                matches += 1
                last_sentence_matches += 1
            else:
                errors += 1
            all_data += 1
            last_sentence_length += 1
    with open(evaluate_file_path, "w") as eval_file:
        print_title_to_file(eval_file, "Part-of-speech Tagging Evaluation")
        print_eval_details(eval_file, model, smoothing, test_file_path, gold_file_path)
        print_title_to_file(eval_file, "sent-num word-accuracy sent-accuracy")
        print_sentence_data(eval_file, sentences_accuracy_list)
        print_macro_avg(eval_file, 1.0 * matches / all_data, sentence_accuracy_for_test_corpus(sentences_accuracy_list))


# run all phases one after another
train_phase(train_file_path, param_file_path)
classify_phase(param_file_path, test_file_path, classification_output_file_path)
evaluate(classification_output_file_path, gold_file_path, evaluate_file_path)