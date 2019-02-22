"""
Remarks about this API:
 - the idea is to provide text input and retrieve it labelled
 - the model is assumed to be already loaded
 - the feature-generation to represent the input before querying the model is responsibility of the tool
 - the input is expected to be a single piece of raw of text, tokenization is not responsibility of the client
 - it is responsibility of the tool to use the same tokenisation (or the most similar one) than the one used for training
"""
import es_core_news_sm
from typing import List, Tuple, Dict

import spacy

from custom_api.feature_generators2 import FeatureGenerator, AffixFeatureGenerator, SpacyBasedFeatureGenerator, DictionaryBasedFeatureGenerator, \
    CapitalizationFeatureGenerator, WordPositionFeatureGenerator
from my_main2 import label_input

FEATURES = []


class CustomDecodingAPI:

    def __init__(self, feature_generators: List[FeatureGenerator]):
        self.feature_generators = feature_generators  # I am not sure what this should look like...

    def deco_input(self, input_text: str):
        model_dir = 'C:\\Users\\agarciap\\Data\\DATASETS\\NCRFpp_tests\\test_with_crf'
        model_name = 'lstmcrf'

        tokens = self.tokenize_input(input_text)
        input_lines = self.compute_features(tokens, separator=' ', fake_label='O')

        decode_results, pred_scores = label_input(input_lines, model_dir, model_name, nbest=5)
        return decode_results, pred_scores

    def tokenize_input(self, input_text: str) -> List[str]:
        # naive, just to test
        return input_text.split(' ')

    def compute_features(self, tokens: List[str], separator: str = '\t', fake_label: str = None) -> List[Tuple[str, Dict[str, str]]]:
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
                line += '{}[{}]{}'.format(separator, feature[0], feature[1][i])
            if fake_label:
                line += separator + fake_label
            featurized_output.append(line.strip() + '\n')

        return featurized_output


if __name__ == '__main__':
    suffix1 = AffixFeatureGenerator(feature_name='SUF1', affix_type='suffix', size=1)
    suffix2 = AffixFeatureGenerator(feature_name='SUF2', affix_type='suffix', size=2)
    suffix3 = AffixFeatureGenerator(feature_name='SUF3', affix_type='suffix', size=3)
    suffix4 = AffixFeatureGenerator(feature_name='SUF4', affix_type='suffix', size=4)
    prefix3 = AffixFeatureGenerator(feature_name='PRE2', affix_type='prefix', size=3)
    prefix4 = AffixFeatureGenerator(feature_name='PRE3', affix_type='prefix', size=4)

    cap = CapitalizationFeatureGenerator(feature_name='CAP', caps_type='first')
    allcaps = CapitalizationFeatureGenerator(feature_name='ALLCAPS', caps_type='all')
    somecaps = CapitalizationFeatureGenerator(feature_name='SOMECAPS', caps_type='some')

    is_first = WordPositionFeatureGenerator(feature_name='FIRST', position_type='first')

    ####
    PATH_TO_CLUSTERS_1 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\berrias50w5600"
    PATH_TO_CLUSTERS_2 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\egunkariaprecleantokc1000p1txt"
    PATH_TO_CLUSTERS_3 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\egunkarias50w5300"
    PATH_TO_CLUSTERS_4 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\egunkariatokpunctlower200"
    PATH_TO_CLUSTERS_5 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\euwikitokpunctlower200bak"
    ####

    brown4 = DictionaryBasedFeatureGenerator(feature_name='BROWN4', input_path=PATH_TO_CLUSTERS_2, unknown_value='0', trim_value_at=4)
    brown6 = DictionaryBasedFeatureGenerator(feature_name='BROWN6', input_path=PATH_TO_CLUSTERS_2, unknown_value='0', trim_value_at=6)
    brown10 = DictionaryBasedFeatureGenerator(feature_name='BROWN10', input_path=PATH_TO_CLUSTERS_2, unknown_value='0', trim_value_at=10)
    brown20 = DictionaryBasedFeatureGenerator(feature_name='BROWN20', input_path=PATH_TO_CLUSTERS_2, unknown_value='0', trim_value_at=20)

    c1 = DictionaryBasedFeatureGenerator(feature_name='C1', input_path=PATH_TO_CLUSTERS_1, unknown_value='0')
    c3 = DictionaryBasedFeatureGenerator(feature_name='C3', input_path=PATH_TO_CLUSTERS_3, unknown_value='0')
    c4 = DictionaryBasedFeatureGenerator(feature_name='C4', input_path=PATH_TO_CLUSTERS_4, unknown_value='0')
    c5 = DictionaryBasedFeatureGenerator(feature_name='C5', input_path=PATH_TO_CLUSTERS_5, unknown_value='0')

    nlp = es_core_news_sm.load()
    spacy_pos = SpacyBasedFeatureGenerator(feature_name='POS', spacy_model=nlp, feature_type='pos')
    spacy_lemma = SpacyBasedFeatureGenerator(feature_name='LEM', spacy_model=nlp, feature_type='lemma')

    custom_deco_api = CustomDecodingAPI(
        feature_generators=[spacy_pos, cap, is_first, allcaps, somecaps, spacy_lemma, prefix3, prefix4, suffix1, suffix2, suffix3, suffix4, brown4,
                            brown6, brown10, brown20, c1, c3, c4, c5])
    generated_features = custom_deco_api.compute_features(tokens='esto es una prueba con palabras bonitas'.split(' '), separator=' ')

    [print(output) for output in generated_features]

    generated_features = custom_deco_api.compute_features(tokens='gaur Donostian denbora ona egiten du'.split(' '), separator=' ')
    [print(output) for output in generated_features]

    deco, probs = custom_deco_api.deco_input('gaur Bilbon denbora ona egiten du eta Pepe Goterak hitz egingo du')
    print(deco)
    print(probs)
