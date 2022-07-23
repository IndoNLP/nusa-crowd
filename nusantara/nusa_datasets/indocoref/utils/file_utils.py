# Taken from https://github.com/valentinakania/indocoref/blob/main/src/utils/file_utils.py
import os
import re
import pickle
import json
import logging
from pathlib import Path
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

class FileUtils:
    @staticmethod
    def write_gold_cluster_to_json(output_dir, name, cluster):
        filename = Path(output_dir).joinpath("gold/" + name).with_suffix(".json")
        FileUtils.write_cluster_to_json(filename, cluster)

    @staticmethod
    def write_mps_result_to_json(output_dir, name, cluster):
        filename = Path(output_dir).joinpath("result-mps/" + name).with_suffix(".json")
        FileUtils.write_cluster_to_json(filename, cluster)

    @staticmethod
    def write_cluster_to_json(filename, cluster):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = {'type': 'clusters', 'clusters': cluster}
        with open(filename, 'w') as f:
            f.write(json.dumps(data))

    @staticmethod
    def write_to_pickle(temp_dir, name, objs):
        filename = Path(temp_dir).joinpath(name).with_suffix(".pkl")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(objs, f)

    @staticmethod
    def read_pickle(temp_dir, name):
        filename = Path(temp_dir).joinpath(name)
        with open(filename, 'rb') as f:
            objs = pickle.load(f)
        return objs

    @staticmethod
    def read_annotated_file(annotated_dir, name):
        with open(Path(annotated_dir).joinpath(name), 'r', encoding='utf-8') as f:
            annotated = f.read()
        sentences = sent_tokenize(annotated)
        return (annotated, sentences)

    @staticmethod
    def read_passage_file(passage_dir, name):
        passage_name = re.sub(r"_[0-9]{8}-[0-9]{6}.+", ".txt", name)
        with open(Path(passage_dir).joinpath(passage_name), 'r', encoding='utf-8') as f:
            passage = f.read()
            sentences = sent_tokenize(passage)
        return sentences