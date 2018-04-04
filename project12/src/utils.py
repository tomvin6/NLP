import re  # for splitting by tabs


def is_line_legal(line):
    return len(line.strip()) > 0 and not line.startswith("#") and len(re.split(r'\t+', line)) == 2


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

