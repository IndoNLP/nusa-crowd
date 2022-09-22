# Taken from https://github.com/valentinakania/indocoref/blob/main/src/utils/feature_utils.py
import re
import nltk

PRONOUNS = ['dia', 'ia', 'beliau', 'mereka', 'kami', 'kita', 'aku', 'saya', 'kamu', 'anda', 'kalian']
PRONOUN_SINGULAR = ['dia', 'ia', 'beliau', 'aku', 'saya', 'kamu', 'anda']
PRONOUN_PLURAL = ['mereka', 'kami', 'kita', 'kalian']
CLITICS = ['mu', 'ku', 'nya']
APPOSITIVES = [',', ';', '(', ')']
COPULATIVES = ['adalah', 'yaitu', 'merupakan', 'ialah', 'yakni']
DEMONSTRATIVES = ['ini', 'itu', 'tersebut']
PERSON = ['orang', 'manusia', 'pria', 'wanita', 'ibu', 'bapak', 'putra', 'putri']
LOCATION = ['tempat', 'lokasi', 'kota', 'provinsi', 'area', 'daerah', 'negara', 'negeri', 'sekitar']
STOPWORDS = [*COPULATIVES, *DEMONSTRATIVES, 'yang', 'itu', 'dan', 'atau', 'tapi', 'nan', 'namun', 'tetapi', 'sang', 'si']

NOUN_POS_TAGS = ["NOUN", "PROPN"]


class FeatureUtils:
    @staticmethod
    def get_chunk_parser():
        grammar = r'''
        NP: {<NUM|DET>* <NOUN|PROPN|PRON>* <.*>* <NOUN|PROPN|PRON>}
            {<NOUN|PROPN|PRON> <NOUN|PROPN|PRON>}
            {<NOUN|PROPN|PRON>* <ADJ>}
            {<NUM|DET>* <NOUN|PROPN|PRON>* <PP>}
            {<NOUN|PROPN|PRON>}
        PP: {<ADP><NOUN|PROPN|PRON>*}
        '''

        chunk_parser = nltk.RegexpParser(grammar)
        return chunk_parser

    @staticmethod
    def find_in_sentence(a, sentences):
        if a.get('sent') is not None and a.get('sent') >= 0:
            return a.get('sent')
        for idx, sent in enumerate(sentences):
            has_all_labels = True
            for l in a['labels']:
                if 'M{}'.format(l) not in sent:
                    has_all_labels = False
                    break
            if has_all_labels:
                return idx
        return 0

    @staticmethod
    def is_pronoun(a):
        return 1 if a.get('text', '').lower() in PRONOUNS or a.get('text', '').lower() in CLITICS else 0

    @staticmethod
    def is_pronoun_singular(a):
        return 1 if a.get('text', '').lower() in PRONOUN_SINGULAR or a.get('text', '').lower() in CLITICS else 0

    @staticmethod
    def is_pronoun_plural(a):
        return 1 if a.get('text', '').lower() in PRONOUN_PLURAL else 0

    @staticmethod
    def is_proper_noun(a):
        return 1 if a.get('pos', '') == 'PROPN' else 0

    @staticmethod
    def find_demonstrative_phrase(a):
        for d in a.get('text').lower().split(' '):
            if d in DEMONSTRATIVES:
                return d
        return None

    @staticmethod
    def strip_demonstrative(a, demonstrative):
        return ' '.join([text for text in a['text'].split(' ') if demonstrative not in text])

    @staticmethod
    def is_ner(a):
        return a.get('ner', 0)

    @staticmethod
    def is_person(a):
        if any((c.lower() in PERSON) for c in a['text']):
            return 1
        return a.get('per', 0)

    @staticmethod
    def is_organization(a):
        return a.get('org', 0)

    @staticmethod
    def is_location(a):
        if any((c.lower() in LOCATION) for c in a['text']):
            return 1
        return a.get('loc', 0)

    @staticmethod
    def is_clitic(a):
        return 1 if a.get('text') in CLITICS else 0

    @staticmethod
    def has_class(a):
        return FeatureUtils.is_person(a) or FeatureUtils.is_organization(a) or FeatureUtils.is_location(a)

    @staticmethod
    def get_head(a):
        tags = a.get('tag', [])
        for t in tags:
            strs, pos = t
            if pos == 'NOUN' or pos == 'PROPN':
                return strs
        return ''

    @staticmethod
    def get_full_head_noun(a):
        head_words = []
        tags = a.get('tag', [])
        for str, pos in tags:
            if pos not in NOUN_POS_TAGS:
                break
            head_words.append(str)
        return ' '.join(head_words)

    @staticmethod
    def get_full_head_proper_noun(a):
        strs = []
        tags = a.get('tag', [])
        for str, pos in tags:
            if pos == 'NOUN' and len(strs) == 0:
                return ''
            if pos == 'PROPN' and str[0].isupper():
                strs.append(str)
        return ' '.join(strs)



class PairFeatureUtils:

    @staticmethod
    def is_same_word_class(a, b):
        return FeatureUtils.is_person(a) and FeatureUtils.is_person(b) or FeatureUtils.is_organization(a) and \
            FeatureUtils.is_organization(b) or FeatureUtils.is_location(a) and FeatureUtils.is_location(b)

    @staticmethod
    def is_word_class_mismatch(a, b):
        return FeatureUtils.has_class(a) and FeatureUtils.has_class(b) and not PairFeatureUtils.is_same_word_class(a, b)
    
    @staticmethod
    def is_exact_match(a, b):
        textA = re.sub('[^0-9a-zA-Z -,.]+', '', a.get('text'))
        textB = re.sub('[^0-9a-zA-Z -,.]+', '', b.get('text'))
        return 1 if textA == textB else 0

    @staticmethod
    def is_name_shortened(a, b):
        textA = re.sub('[^0-9a-zA-Z -,.]+', '', a.get('text').lower())
        textB = re.sub('[^0-9a-zA-Z -,.]+', '', b.get('text').lower())
        if textA == '' or textB == '':
            return 0
        return 1 if (textA in textB or textB in textA) \
            and FeatureUtils.is_ner(a) and FeatureUtils.is_ner(b) and PairFeatureUtils.is_same_word_class(a, b) else 0

    @staticmethod
    def is_appositive(a, b, sentences):
        idxA = FeatureUtils.find_in_sentence(a, sentences)
        idxB = FeatureUtils.find_in_sentence(b, sentences)
        if idxA != idxB:
            return 0
        if idxA >= len(sentences):
            token = len(' '.join(sentences).split('.'))
            idxA -= (token - len(sentences))
        try:
            sent = sentences[idxA]
            start = min(sent.find(a.get('text')), sent.find(b.get('text'))) + len(a.get('text'))
            end = max(sent.find(a.get('text')), sent.find(b.get('text')))
        except:
            return 0
        if sent[start:end].count(' ') > 1:
            return 0
        if any((c.lower() in APPOSITIVES) for c in sent[start:end]):
            return FeatureUtils.is_ner(a) ^ FeatureUtils.is_ner(b)
        return 0

    @staticmethod
    def is_copulative(a, b, sentences):
        idxA = FeatureUtils.find_in_sentence(a, sentences)
        idxB = FeatureUtils.find_in_sentence(b, sentences)
        if idxA != idxB:
            return 0
        if idxA >= len(sentences):
            token = len(' '.join(sentences).split('.'))
            idxA -= (token - len(sentences))
        try:
            sent = sentences[idxA]
            start = min(sent.find(a.get('text')), sent.find(b.get('text'))) + len(a.get('text'))
            end = max(sent.find(a.get('text')), sent.find(b.get('text')))
        except:
            return 0
        return 1 if any((c.lower() in COPULATIVES) for c in sent[start:end].split()) else 0

    @staticmethod
    def is_demonstrative(a, b):
        d = FeatureUtils.find_demonstrative_phrase(b)
        if not d:
            return 0
        cleaned_b = FeatureUtils.strip_demonstrative(b, d)
        for c in cleaned_b.lower().split(' '):
            if c in STOPWORDS:
                continue
            if c not in a['text'].lower().split(' '):
                return 0
        return 1

    @staticmethod
    def is_abbreviation(a, b):
        if a['text'].lower() == b['text'].lower():
            return 0
        short = a.get('text') if len(a) < len(b) else b.get('text')
        long = a.get('text') if len(a) >= len(b) else b.get('text')
        short = short.replace('.', '')
        longs = long.split(' ')
        abbreviation = ''.join([c[0] for c in longs if len(c) > 1])
        if abbreviation.lower() == short.lower():
            return 1
        return 0


    @staticmethod
    def is_relaxed_match(a, b):
        if a['text'].lower() == b['text'].lower():
            return 0
        short = a if len(a['text']) < len(b['text']) else b
        long = a if len(a['text']) >= len(b['text']) else b
        short_full_head = FeatureUtils.get_full_head_noun(short).lower()
        long_full_head = FeatureUtils.get_full_head_noun(long).lower()
        for c in short_full_head.split(' '):
            if c not in long_full_head.split(' '):
                return 0
        return 1 if PairFeatureUtils.is_same_word_class(a, b) else 0
        
    @staticmethod
    def is_head_match(a, b):
        if FeatureUtils.get_head(a).lower() == FeatureUtils.get_head(b).lower():
            return 1
        return 0

    @staticmethod
    def is_full_proper_head_match(a, b):
        a_propn_head = FeatureUtils.get_full_head_proper_noun(a).lower()
        b_propn_head = FeatureUtils.get_full_head_proper_noun(b).lower()
        if a_propn_head == '' or b_propn_head == '':
            return 0
        short_head = a_propn_head if len(a_propn_head) < len(b_propn_head) else b_propn_head
        long_head = a_propn_head if len(a_propn_head) >= len(b_propn_head) else b_propn_head
        for word in short_head.split(' '):
            if word not in long_head.split(' '):
                return 0
        return 1