from src.first_order_classifier import *
from src.basic_classifier import *
from src.constants import *

model = int(sys.argv[1])
train_file_path = sys.argv[2]  # path for train file
smoothing = sys.argv[3]

# assign default values
if str(smoothing).lower() == "t" or str(smoothing).lower() == "yes":
    smoothing = True

# execute train by model argument
if model == 1:
    train_phase(train_file_path, param_file_path)
else:
    model = 2
    train_first_order_classifier(train_file_path, lexical_file_path, structural_file_path, model, smoothing)