"""
Better to use classes as feature generators so the code can be reused with different parameters

"""
from typing import List, Tuple

from spacy.tokens.doc import Doc


class AffixFeatureGenerator:

    def __init__(self, feature_name: str, affix_type: str, size: int):
        """
        :param feature_name:
        :param affix_type: 'suffix' or 'affix'
        :param size:
        """
        self.feature_name = feature_name
        self.affix_type = affix_type
        self.size = size

    def generate_feature(self, tokens: List[str]) -> Tuple[str, List[str]]:
        if self.affix_type == 'suffix':
            feature_values = [tok[-self.size:] if len(tok) > self.size else tok for tok in tokens]
        elif self.affix_type == 'prefix':
            feature_values = [tok[:self.size] if len(tok) > self.size else tok for tok in tokens]
        else:
            raise Exception('Incorrect affix type: {}'.format(self.affix_type))
        return self.feature_name, feature_values


class DictionaryBasedFeatureGenerator:

    def __init__(self, feature_name, input_path, unknown_value: str = '-1', values_separator: str = ' ', trim_value_at: int = None):
        self.feature_name = feature_name
        self.feat_dict = self.load(input_path)
        self.unknown_value = unknown_value
        self.values_separator = values_separator
        self.trim_value_at = trim_value_at

    def load(self, input_path):
        pairs = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip().split(self.values_separator)) == 2:
                    word, feat = line.strip().split(self.values_separator)
                    pairs[word] = feat
        return pairs

    def generate_feature(self, tokens: List[str]) -> Tuple[str, List[str]]:
        feature_values = [self.feat_dict.get(tok, self.unknown_value) for tok in tokens]
        if self.trim_value_at is not None:
            feature_values = [val[:self.trim_value_at] if len(val) > self.trim_value_at else val for val in feature_values]
        return self.feature_name, feature_values


class SpacyBasedFeatureGenerator:

    def __init__(self, feature_name, spacy_model, feature_type: str):
        """
        :param feature_name:
        :param spacy_model:
        :param feature_type: 'pos' or 'lemma' (for now)
        """
        self.feature_name = feature_name
        self.spacy_model = spacy_model
        self.feature_type = feature_type

    def generate_feature(self, tokens: List[str]) -> Tuple[str, List[str]]:
        doc = Doc(self.spacy_model.vocab, words=tokens)
        doc = self.spacy_model(doc.text)
        if self.feature_type == 'pos':
            feature_values = [tok.pos_ for tok in doc]
        elif self.feature_type == 'lemma':
            feature_values = [tok.lemma_ for tok in doc]
        else:
            raise Exception('Incorrect feature type: {}'.format(self.feature_type))
        return self.feature_name, feature_values
