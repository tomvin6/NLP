from src.first_order_classifier import *
from src.basic_classifier import *
from src.constants import *

model = int(sys.argv[1])
test_file_path = sys.argv[2]  # path for test file

# execute train by model argument
if model == 1:
    classify_phase(get_trained_model_file, param_file_path, test_file_path, classification_output_file_path, classify_basic)
else:
    model = 2
    lexical_file_path = sys.argv[3]  # according to spec, we expect to get param-file 1 as arg
    structural_file_path = sys.argv[4]  # according to spec, we expect to get param-file 1 as arg
    possible_tags = load_possible_tags(structural_file_path)
    conf_matrix = init_conf_matrix(possible_tags)
    decode(test_file_path, get_trained_model(structural_file_path, lexical_file_path), possible_tags, class_output_path)



