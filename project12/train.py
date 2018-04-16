from src.first_order_classifier import *
from src.basic_classifier import *
from src.constants import *

model = int(sys.argv[1])

train_file_path = ""
# take args of use default
if len(sys.argv) > 3:
    train_file_path = sys.argv[2]  # path for train file
    smoothing = sys.argv[3]
else:
    smoothing = "y"
    train_file_path = default_train_file_path
    print "default values chosen for train file: " + train_file_path + " smoothing = " + smoothing

# assign default values
if str(smoothing).lower() == "t" or str(smoothing).lower() == "yes":
    smoothing = True

# execute train by model argument
if model == 1:
    train_phase(train_file_path, default_param_file_path)
else:
    model = 2
    train_first_order_classifier(train_file_path, default_lexical_file_path, default_structural_file_path, model, smoothing)