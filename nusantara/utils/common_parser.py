from typing import Iterable

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


def load_ud_data(filepath, filter_kwargs=None, assert_fn=None):
    """
    Load and parse conllu data.

    Proposed by @fhudi for issue #34 and #9.

    :param filepath: file path
    :param filter_kwargs: filtering tokens, see conllu.models.TokenList.filter()
    :param assert_fn: assertion to make sure raw data is in the expected format
    :return: generator with schema following CONLLU
    """
    dataset_raw = parse(open(filepath).read())

    filter_kwargs = filter_kwargs or dict()
    if callable(assert_fn):
        for token_list in dataset_raw:
            assert_fn(token_list)

    return map(lambda sent: {**sent.metadata, **pd.DataFrame(sent.filter(**filter_kwargs)).to_dict(orient="list")}, dataset_raw)


def load_ud_data_as_nusantara_kb(filepath, dataset_source: Iterable = tuple()):
    """
    Load and parse conllu data, followed by mapping its elements to Nusantara Knowledge Base schema.

    Proposed by @fhudi for issue #34 and #9.

    :param filepath: file path
    :param dataset_source: dataset with source schema (output of load_ud_data())
    :return: generator for Nusantara KB schema
    """
    dataset_source = dataset_source or list(load_ud_data(filepath))

    def as_nusa_kb(tokens):
        sent_id = tokens["sent_id"]
        offsets = get_span_offsets(tokens["form"], tokens["text"])
        return {
            "id": sent_id,
            "passages": [
                {
                    "id": f"{sent_id}_passages",
                    "type": "",
                    "text": [tokens["text"]],
                    "offsets": [(0, len(tokens["text"]))],
                }
            ],
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

    try:
        iterchar = iter(iter_char())
        span_len, cur_char = next(iterchar)
        for offset, j in enumerate(text_concatenated):
            if cur_char != j:
                if j in delimiters:
                    continue
                else:
                    raise AssertionError(f"Char '{j}' at pos {offset} does not match char '{cur_char}' from span #{span_idx} ('{span}'), and is not in delimiters {delimiters};")
            else:
                if span_len is not None:
                    offsets.append((offset, offset + span_len))
                span_len, cur_char = next(iterchar)
        raise AssertionError("Text is too short, not enough character for whole spans.")

    except StopIteration:
        return offsets
