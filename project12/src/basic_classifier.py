import sys
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
    return trained_model.get(segment, "NNP")


def get_trained_model(param_path):
    with open(param_path, "r") as model_output:
        return json.loads(model_output.read())


# run all phases one after another
train_phase(train_file_path, param_file_path)
classify_phase(get_trained_model, param_file_path, test_file_path, classification_output_file_path, classify_basic)
evaluate(classification_output_file_path, gold_file_path, evaluate_file_path, test_file_path, model, smoothing)