from src.first_order_classifier import *
from src.basic_classifier import *
from src.constants import *

# TODO: GET PATHS FROM COMMAND LINE OR USE DEFAULT
tagged_file = sys.argv[1]
gold_file_path = sys.argv[2]
model = int(sys.argv[3])
smoothing = sys.argv[4]

# assign default values
if model != 1:
    model = 2
if str(smoothing).lower() == "t" or str(smoothing).lower() == "yes":
    smoothing = True

if model == 1:
    evaluate(classification_output_file_path, gold_file_path, evaluate_file_path, "heb-pos.test", model, smoothing)
else:
    conf_matrix = evaluate(class_output_path, gold_file_path, evaluate_file_path, "heb-pos.test", 2, smoothing)



