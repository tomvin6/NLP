from src.first_order_classifier import *
from src.basic_classifier import *
from src.constants import *

model = int(sys.argv[1])
train_file_path = sys.argv[2]  # path for train file
smoothing = sys.argv[3]
# TODO: GET PATHS FROM COMMAND LINE OR USE DEFAULT

# assign default values
if model != 1:
    model = 2
if str(smoothing).lower() == "t" or str(smoothing).lower() == "yes":
    smoothing = True


if model == 1:
    train_phase(train_file_path, param_file_path)
else:
    train_first_order_classifier(train_file_path, lexical_file_path, structural_file_path, model, smoothing)
