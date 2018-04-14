from src.first_order_classifier import *
from src.basic_classifier import *
from src.constants import *

model = int(sys.argv[1])
test_file_path = sys.argv[2]  # path for train file

# TODO: GET PATHS FROM COMMAND LINE OR USE DEFAULT

# assign default values
if model != 1:
    model = 2

if model == 1:
    classify_phase(get_trained_model_file, param_file_path, test_file_path, classification_output_file_path, classify_basic)
else:
    possible_tags = load_possible_tags(structural_file_path)
    conf_matrix = init_conf_matrix(possible_tags)
    decode(test_file_path, get_trained_model(structural_file_path, lexical_file_path), possible_tags, class_output_path)



