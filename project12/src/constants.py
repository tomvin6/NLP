import os  # for file separator (in linux & windows)

# tmp parameter files:
contants_file_path = os.path.dirname(os.path.realpath(__file__))
root_project_folder = os.path.join(contants_file_path, "..")
default_lexical_file_path = os.path.join(root_project_folder, 'exps', 'lexical-model.txt')  # train phase output
default_structural_file_path = os.path.join(root_project_folder, 'exps', 'structural-model.txt')  # train phase output
default_param_file_path = os.path.join(root_project_folder, 'exps', 'train-param-file.txt')  # train phase output
classification_output_file_path = os.path.join(root_project_folder, 'results', 'classifications.tagged')  # decode phase output
default_train_file_path = os.path.join(root_project_folder, 'exps', 'heb-pos.train')

class_output_path = os.path.join(root_project_folder, 'results', "classifications-bigram.tagged")  # decode phase output
evaluate_file_path = os.path.join(root_project_folder, 'results', 'evaluations.eval') # evaluate phase output


