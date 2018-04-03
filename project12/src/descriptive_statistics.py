import sys
import re  # for splitting by tabs
import os  # for file separator (in linux & windows)

tagged = sys.argv[1]  # 1 - majority, 2 - bi-gram
fileName = sys.argv[2]  # 1 - majority, 2 - bi-gram
model = sys.argv[3]
smoothing = sys.argv[4]

# get file as object
gold_file = open('..' + os.sep + 'exps' + os.sep + fileName)
train_file = open('..' + os.sep + 'exps' + os.sep + 'heb-pos.train')


def descriptive_statistics():
    uni_count = 0
    segment_words = set()
    segment_tags = set()
    words_tag_dict = dict()
    train_lines = train_file.readlines()
    gold_lines = gold_file.readlines()
    # count train file
    for line in train_lines:
        uni_count += 1
        if len(line.strip('\n')) > 0 and not line.startswith("#"):
            tokens = re.split(r'\t+', line)
            segment_words.add(tokens[0])
            segment_tags.add(tokens[1])
            # calc average tags per word
            if words_tag_dict.has_key(tokens[0]):
                words_tag_dict[tokens[0]].add(tokens[1])
            else:
                words_tag_dict[tokens[0]] = set()
                words_tag_dict[tokens[0]].add(tokens[1])

    # count gold file
    for line in gold_lines:
        uni_count += 1
        if len(line.strip('\n')) > 0 and not line.startswith("#"):
            tokens = re.split(r'\t+', line)
            segment_words.add(tokens[0])
            segment_tags.add(tokens[1])
            # calc average tags per word
            if words_tag_dict.has_key(tokens[0]):
                words_tag_dict[tokens[0]].add(tokens[1])
            else:
                words_tag_dict[tokens[0]] = set()
                words_tag_dict[tokens[0]].add(tokens[1])

    print "word type count = " + str(len(segment_words))
    print "word count = " + str(uni_count)
    print "tag count = " + str(len(segment_tags))
    # get average tags per word segment
    total_different_tags = 0
    total_words = 0
    for tags in words_tag_dict.values():
        total_different_tags += len(tags)
        total_words += 1
    print "average tags per word = " + str(1.0 * total_different_tags / total_words)


descriptive_statistics()