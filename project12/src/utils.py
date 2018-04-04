import re  # for splitting by tabs

def is_line_legal(line):
    return len(line.strip()) > 0 and not line.startswith("#")


def load_key_value_file_as_list(file_path):
    output_list = []
    with open(file_path, "r") as file_object:
        for line in file_object.readlines():
            if is_line_legal(line):
                tokens = re.split(r'\t+', line)
                output_list.append((tokens[0].strip(), tokens[1].strip()))
    return output_list
