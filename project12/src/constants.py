import os  # for file separator (in linux & windows)

# tmp parameter files:
lexical_file_path = 'exps' + os.sep + 'lexical-model.txt'  # train phase output
structural_file_path = 'exps' + os.sep + 'structural-model.txt'  # train phase output
param_file_path = 'exps' + os.sep + 'train-param-file.txt'  # train phase output
classification_output_file_path = 'results' + os.sep + 'classifications.tagged'  # decode phase output