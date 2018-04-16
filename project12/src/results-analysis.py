from first_order_classifier import *
from constants import *


def file_len(f_path):
    return sum(1 for line in open(f_path))


# method to split train file to files:
def generate_x_lines_file(orig_file_path, output_file_path, percent_to_read):
    num_of_lines = file_len(orig_file_path)
    lines_to_read = int(percent_to_read * num_of_lines)
    with open(output_file_path, "w") as output:
        f = open(orig_file_path)
        for i in range(lines_to_read):
            line = f.next()
            output.write(line)
        f.close()


# for running all phases & getting sorted errors
# train_first_order_classifier(train_file_path, lexical_file_path, structural_file_path, 2)
# possible_tags = load_possible_tags(structural_file_path)
# conf_matrix = init_conf_matrix(possible_tags)
# decode(test_file_path, get_trained_model(structural_file_path, lexical_file_path), possible_tags, class_output_path)
# conf_matrix = evaluate(class_output_path, gold_file_path, evaluate_file_path, test_file_path, conf_matrix, 2, smoothing)
# sorted_errors = get_sorted_errors(conf_matrix)


# dividing the train to 10 parts experiment, measure accuracy:
tmp_train_file = '..' + os.sep + 'tmp' + os.sep + 'partial-train.txt'

for i in range(1, 11):
    generate_x_lines_file(default_train_file_path, tmp_train_file, i * 0.10)
    train_first_order_classifier(tmp_train_file, default_lexical_file_path, default_structural_file_path, 2, True)
    possible_tags = load_possible_tags(default_structural_file_path)
    conf_matrix = init_conf_matrix(possible_tags)
    decode(test_file_path, get_trained_model(default_structural_file_path, default_lexical_file_path), possible_tags, class_output_path)
    conf_matrix = evaluate(class_output_path, gold_file_path, evaluate_file_path + str(i), test_file_path, 2, True, conf_matrix)
    sorted_errors = get_sorted_errors(conf_matrix)
    print "generated output " + str(i)