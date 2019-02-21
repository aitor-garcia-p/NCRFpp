"""
Remarks about this API:
 - the idea is to provide text input and retrieve it labelled
 - the model is assumed to be already loaded
 - the feature-generation to represent the input before querying the model is responsibility of the tool
 - the input is expected to be a single piece of raw of text, tokenization is not responsibility of the client
 - it is responsibility of the tool to use the same tokenisation (or the most similar one) than the one used for training
"""
from typing import List, Tuple, Dict

import spacy

from custom_api.feature_generators2 import FeatureGenerator, AffixFeatureGenerator, SpacyBasedFeatureGenerator

FEATURES = []


class CustomDecodingAPI:

    def __init__(self, feature_generators: List[FeatureGenerator]):
        self.feature_generators = feature_generators  # I am not sure what this should look like...
        pass

    @staticmethod
    def deco_input(input_text: str):
        pass

    @staticmethod
    def tokenize_input(input_text: str) -> List[str]:
        pass

    def compute_features(self, tokens: List[str], separator: str = '\t') -> List[Tuple[str, Dict[str, str]]]:
        """
        Receives a list of tokens, and computes the features for each of them (where are those features specified?)
        The result is a list of tuples, containing each token and a dictionary of features
        :param separator:
        :param tokens:
        :return:
        """
        all_features = []
        for feature_generator in self.feature_generators:
            features = feature_generator.generate_feature(tokens)
            all_features.append(features)

        featurized_output = []
        for i, token in enumerate(tokens):
            line = token
            for feature in all_features:
                line += '{}[{}]{}'.format(separator, feature[0], features[1][i])
            featurized_output.append(line.strip() + '\n')

        print(featurized_output)
        # not really necessary...
        # some features might need the whole text to be computed (e.g. part-of-speech)
        # so better that the feature generators admit full texts (sentences or whatever)
        # and internally assign the feature individually or in group

        pass


if __name__ == '__main__':
    suffix2 = AffixFeatureGenerator(feature_name='suffix2', affix_type='suffix', size=2)
    suffix3 = AffixFeatureGenerator(feature_name='suffix3', affix_type='suffix', size=3)
    prefix2 = AffixFeatureGenerator(feature_name='prefix2', affix_type='prefix', size=2)
    prefix3 = AffixFeatureGenerator(feature_name='prefix3', affix_type='prefix', size=3)

    nlp = es_core_news_sm.load() # spacy.load('es_core_news_sm')
    spacy_pos = SpacyBasedFeatureGenerator(feature_name='pos', spacy_model=nlp, feature_type='pos')
    spacy_lemma = SpacyBasedFeatureGenerator(feature_name='lemma', spacy_model=nlp, feature_type='lemma')

    custom_deco_api = CustomDecodingAPI(feature_generators=[suffix2, suffix3, prefix2, prefix3, spacy_pos, spacy_lemma])
    custom_deco_api.compute_features(tokens='esto es una prueba con palabras bonitas'.split(' '))
