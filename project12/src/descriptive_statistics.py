import sys
import re  # for splitting by tabs
import os  # for file separator (in linux & windows)

from utils import *

tagged = sys.argv[1]  # 1 - majority, 2 - bi-gram
fileName = sys.argv[2]  # 1 - majority, 2 - bi-gram
model = sys.argv[3]
smoothing = sys.argv[4]

# get file as object
gold_file = open('..' + os.sep + 'exps' + os.sep + fileName)
train_file = open('..' + os.sep + 'exps' + os.sep + 'heb-pos.train')


def count_file(file_lines, words_tag_dict, segments_set, tags_set, prev_counter):
    counter = prev_counter
    for line in file_lines:
        if is_comment_line(line) or end_of_sentence(line):
            continue
        if is_line_legal(line):
            counter += 1
            tokens = re.split(r'\t+', line)
            segments_set.add(tokens[0])
            tags_set.add(tokens[1])
            # calc average tags per word
            if words_tag_dict.has_key(tokens[0]):
                words_tag_dict[tokens[0]].add(tokens[1])
            else:
                words_tag_dict[tokens[0]] = set()
                words_tag_dict[tokens[0]].add(tokens[1])
    return counter


# get average tags per word segment
def get_average(words_tag_dict):
    total_different_tags = 0
    total_words = 0
    for tags in words_tag_dict.values():
        total_different_tags += len(tags)
        total_words += 1
    return 1.0 * total_different_tags / total_words


def descriptive_statistics():
    segment_words = set()
    gold_segment_words = set()
    segment_tags = set()
    gold_segment_tags = set()
    words_tag_dict = dict()
    gold_words_tag_dict = dict()
    train_lines = train_file.readlines()
    gold_lines = gold_file.readlines()
    # count train file
    line_count = count_file(train_lines, words_tag_dict, segment_words, segment_tags, 0)
    print "train file: "
    print "     word count = " + str(line_count)
    print "     word type count = " + str(len(segment_words))
    print "     tags type count = " + str(len(segment_tags))
    print "     ambi value:  = " + str(get_average(words_tag_dict))

    # count gold file
    gold_line_count = count_file(gold_lines, gold_words_tag_dict, gold_segment_words, gold_segment_tags, 0)
    print "gold file: "
    print "     word count = " + str(gold_line_count)
    print "     word type count = " + str(len(gold_segment_words))
    print "     tags type count = " + str(len(gold_segment_tags))
    print "     ambi value:  = " + str(get_average(gold_words_tag_dict))

    # merged results
    total_line_counter = count_file(gold_lines, words_tag_dict, segment_words, segment_tags, line_count)
    print "both files: "
    print "     word count = " + str(total_line_counter)
    print "     word type count = " + str(len(segment_words))
    print "     tags type count = " + str(len(segment_tags))
    print "     ambi value:  = " + str(get_average(words_tag_dict))



descriptive_statistics()