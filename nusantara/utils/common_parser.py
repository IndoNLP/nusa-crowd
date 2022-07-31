import pandas as pd
from conllu import parse


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


def load_ud_data(filepath):
    """
    Load and parse conllu data.

    Proposed by @fhudi for issue #xxx and #xxx.

    :param filepath: file path
    :return: generator with schema following CONLLU
    """
    dataset_raw = parse(open(filepath).read())
    return map(lambda sent: {**sent.metadata, **pd.DataFrame(sent).to_dict(orient="list")}, dataset_raw)


def load_ud_data_as_nusantara_kb(filepath):
    """
    Load and parse conllu data, followed by mapping its elements to Nusantara Knowledge Base schema.

    Proposed by @fhudi for issue #xxx and #xxx.

    :param filepath: file path
    :return: generator for Nusantara KB schema
    """
    dataset_source = list(load_ud_data(filepath))

    def as_nusa_kb(tokens):
        sent_id = tokens["sent_id"]
        return {
            "id": sent_id,
            "entities": [
                {
                    "id": f"{sent_id}_EntID_{ent_id}",
                    "type": ent_type,
                    "text": [ent_text],
                    "offsets": [],
                    "normalized": [
                        {
                            "db_name": norm_text,
                            "db_id": None,
                        }
                    ],
                }
                for (ent_id, ent_type, ent_text, norm_text) in zip(tokens["id"], tokens["upos"], tokens["form"], tokens["lemma"])
            ],
            "relations": [
                {
                    "id": f"{sent_id}_RelID_{rel_id}",
                    "type": rel_type,
                    "arg1_id": f"{sent_id}_EntID_{rel_child}",
                    "arg2_id": f"{sent_id}_EntID_{rel_parent}",
                    "normalized": [],
                }
                for rel_id, (rel_type, rel_child, rel_parent) in enumerate(zip(tokens["deprel"], tokens["id"], tokens["head"]))
            ],
            "events": [],
            "passages": [],
            "coreferences": [],
        }

    return map(as_nusa_kb, dataset_source)
