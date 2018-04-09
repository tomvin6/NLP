import re  # for splitting by tabs

# optional parameter
MISSING = object()


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
    file_to_print.write("# Model: " + model + "\n")
    file_to_print.write("# Smoothing: " + smoothing + "\n")
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
    cells = lex_line.split("\t")
    return ', '.join(cells[1:]), cells[:1]


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
