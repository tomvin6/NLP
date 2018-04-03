import sys
import re  # for splitting by tabs
import os  # for file separator (in linux & windows)
import json  # for writing the parameter file
tagged = sys.argv[1]  # 1 - majority, 2 - bi-gram
fileName = sys.argv[2]  # 1 - majority, 2 - bi-gram
model = sys.argv[3]
smoothing = sys.argv[4]

# file paths
param_file_path = 'param-file1.txt'
train_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.train'
test_file_path = '..' + os.sep + 'exps' + os.sep + 'heb-pos.test'
classification_output_file_path = 'classifications.txt'


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
        uni_count += 1
        if len(line.strip()) > 0 and not line.startswith("#"):
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
        if len(line.strip()) > 0 and not line.startswith("#"):
            segment = line.strip('\n')
            classifications.append((segment, trained_model.get(segment, "NNP")))
        else:
            classifications.append(("", ""))  # for saving the spaces between sentences in classification file
    # write classifications to file
    with open(classification_path, "w") as classification_file:
        for (segment, classification) in classifications:
            classification_file.write(segment + '\t' + classification + '\n')


train_phase(train_file_path, param_file_path)
classify_phase(param_file_path, test_file_path, classification_output_file_path)