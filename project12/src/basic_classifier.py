import sys
import os  # for file separator (in linux & windows)
import json  # for writing the parameter file
from utils import *


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
    words_tag_model_output = dict()
    words_tag_model_tmp_data = build_segment_tags_map(train_file_path)
    # filter only majority tag for each word segment
    for (segment, tags_map) in words_tag_model_tmp_data.items():
        words_tag_model_output[segment] = get_majority_tag(tags_map)
    # writing the output as parameter file
    with open(param_file_path_output, "w") as model_output:
        model_output.write(json.dumps(words_tag_model_output))


def classify_basic(segment, trained_model):
    return trained_model.get(segment, NNP_STATE)


def get_trained_model_file(param_path):
    with open(param_path, "r") as model_output:
        return json.loads(model_output.read())


def classify_phase(get_trained_model, parameters, test_path, classification_path, classifier=classify_basic):
    classifications = []
    # get trained model from parameters file
    trained_model = get_trained_model(parameters)
    # get test file data
    with open(test_path, "r") as test_data:
        test_lines = test_data.readlines()
    for line in test_lines:
        if is_comment_line(line):
            continue
        if not end_of_sentence(line):
            segment = line.strip('\n')
            classifications.append((segment, classifier(segment, trained_model)))
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