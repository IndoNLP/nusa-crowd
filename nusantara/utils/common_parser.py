def load_conll_data(file_path):
    # Read file
    data = open(file_path, "r").readlines()

    # Prepare buffer
    dataset = []
    sentence, seq_label = [], []
    for line in data:
        if len(line.strip()) > 0:
            token, label = line[:-1].split("\t")
            sentence.append(token)
            seq_label.append(label)
        else:
            dataset.append({"sentence": sentence, "label": seq_label})
            sentence = []
            seq_label = []
    return dataset
