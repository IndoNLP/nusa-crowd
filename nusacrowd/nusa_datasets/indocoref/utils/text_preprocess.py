# Taken from https://github.com/valentinakania/indocoref/blob/main/src/utils/text_preprocess.py
# with modification to handle nested annotations in a better way and provide char index offset
import os
import re
import logging

from nusacrowd.nusa_datasets.indocoref.utils.feature_utils import FeatureUtils, CLITICS
from nusacrowd.nusa_datasets.indocoref.utils.file_utils import FileUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TextPreprocess:
    def __init__(self, annotated_dir):
        files = os.listdir(annotated_dir)
        self.filenames = files
        self.annotated_dir = annotated_dir

    def run(self, log_step=10):
        total_files = len(self.filenames)
        mentions_per_file = {}
        for idx, name in enumerate(self.filenames):
            if log_step != 0:
                if (idx + 1) % log_step == 0:
                    logging.info("Preprocessing %d/%d" %
                                 (idx + 1, total_files))
            annotation, sents = FileUtils.read_annotated_file(
                self.annotated_dir, name)

            mentions = self.tokenize_by_regex(annotation)
            self.gen_mention_attributes(mentions, sents)
            mentions_per_file[name] = mentions
        return mentions_per_file

    # Tokenize annotated document by regex according to SACR format
    # SACR format:
    # [...<other text>..., '{', <ID>, ':', <property_name>, '=', "<property_value>" <annotated_mention>, '}', ...<other text>...]
    # Ex:
    ### ['{', 'M1', ':', 'jenis', '=', '"noun phrase other" judul novel', '}', 'dari tahun 1990.']
    def tokenize_by_regex(self, annotation):
        tokens = re.split(r'({|:|=|})', annotation)
        i = 0
        mention_id = 0
        current_offset = 0
        mentions = []
        stack = []

        while i < len(tokens):
            if tokens[i] == '}':
                stack_top = stack.pop()
                if stack:
                    stack[-1]['text'] += stack_top['text']
                
                mention = {}
                mention['labels'] = stack_top['labels']
                mention['num_labels'] = stack_top['num_labels']
                mention['text'] = stack_top['text']
                mention['class'] = stack_top['class']
                mention['offset'] = (stack_top['offset'], current_offset)

                if stack and (stack[-1]['class'] == '' or mention['class'] == ''):
                    stack[-1]['class'] = stack[-1]['class'] or mention['class']
                    stack[-1]['text'] = stack[-1]['text'] or mention['text']
                    stack[-1]['labels'] += mention['labels']
                else:
                    mention_id += 1
                    mention['id'] = mention_id
                    mentions.append(mention)
            elif tokens[i] == '{':
                i += 1
                label = tokens[i][1:]

                # Skipping ':', <property_name>, '='
                i += 4

                found = re.search(r'^"([^"]*)" (.*)$', tokens[i])
                class_type = found.group(1)
                text = found.group(2)

                stack.append({
                    'labels': [label],
                    'num_labels': len(stack),
                    'offset': current_offset,
                    'class': class_type,
                    'text': text
                })
                current_offset += len(text)
            else:
                if stack:
                    stack[-1]['text'] += tokens[i]
                current_offset += len(tokens[i])
            i += 1
        return mentions

    def gen_mention_attributes(self, mentions, sentences):
        for idx, mention in enumerate(mentions):
            mention['pronoun'] = FeatureUtils.is_pronoun(mention)
            mention['proper'] = FeatureUtils.is_proper_noun(mention)
            mention['sent'] = FeatureUtils.find_in_sentence(mention, sentences)
            mention['cluster'] = idx
            mention['per'] = 1 if mention['class'] == 'named-entity person' else 0
            mention['org'] = 1 if mention['class'] == 'named-entity organisation' else 0
            mention['loc'] = 1 if mention['class'] == 'named-entity place' else 0
            mention['ner'] = 1 if 'named-entity' in mention['class'] else 0
