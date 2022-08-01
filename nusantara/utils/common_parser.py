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

    Proposed by @fhudi for issue #34 and #9.

    :param filepath: file path
    :return: generator with schema following CONLLU
    """
    dataset_raw = parse(open(filepath).read())
    return map(lambda sent: {**sent.metadata, **pd.DataFrame(sent).to_dict(orient="list")}, dataset_raw)


def load_ud_data_as_nusantara_kb(filepath):
    """
    Load and parse conllu data, followed by mapping its elements to Nusantara Knowledge Base schema.

    Proposed by @fhudi for issue #34 and #9.

    :param filepath: file path
    :return: generator for Nusantara KB schema
    """
    dataset_source = list(load_ud_data(filepath))

    def as_nusa_kb(tokens):
        sent_id = tokens["sent_id"]
        offsets = get_span_offsets(tokens["form"], tokens["text"])
        return {
            "id": sent_id,
            "entities": [
                {
                    "id": f"{sent_id}_EntID_{ent_id}",
                    "type": ent_type,
                    "text": [ent_text],
                    "offsets": [offset],
                    "normalized": [
                        {
                            "db_name": norm_text,
                            "db_id": None,
                        }
                    ],
                }
                for (ent_id, ent_type, ent_text, offset, norm_text) in zip(tokens["id"], tokens["upos"], tokens["form"], offsets, tokens["lemma"])
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


def get_span_offsets(spans_inorder, text_concatenated, delimiters={" "}):
    """
    Getting the offset of each span assuming spans_inorder is retrieved by splitting text_concatenated using one of delimiters.

    :param spans_inorder: Iterable<String>
    :param text_concatenated: String
    :param delimiters: Set<char>
    :return: List of pair (lo, hi) indicating the start index (inclusive) and end index (exclusive) of original text, respectively.
    """
    offsets = []
    span_idx, span = None, None

    def iter_char():
        nonlocal span_idx, span
        for span_idx, span in enumerate(spans_inorder):
            for st, ch in enumerate(span):
                yield len(span) if st == 0 else None, ch

    iterchar = iter(iter_char())
    try:
        span_len, cur_char = next(iterchar)
    except StopIteration:
        return offsets

    for offset, j in enumerate(text_concatenated):
        if cur_char != j:
            if j in delimiters:
                continue
            else:
                raise AssertionError(f"Char '{j}' at pos {offset} does not match char '{cur_char}' from span #{span_idx} ('{span}'), and is not in delimiters {delimiters};")
        else:
            if span_len is not None:
                offsets.append((offset, offset + span_len))
            try:
                span_len, cur_char = next(iterchar)
            except StopIteration:
                return offsets
