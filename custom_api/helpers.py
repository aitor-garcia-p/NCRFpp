from typing import List, Tuple, Dict

from custom_api.feature_generators import FeatureGenerator

FAKE_LABEL='O'

def compute_features(tokens: List[str], feature_generators: List[FeatureGenerator], separator: str = '\t', labels: List[str] = None) \
        -> List[Tuple[str, Dict[str, str]]]:
    """
    Receives a list of tokens, and computes the features for each of them (where are those features specified?)
    The result is a list of tuples, containing each token and a dictionary of features.
    Labels are the provided gold labels (same number as tokens). If missing (for inference) a fake 'O' label is added.
    :param feature_generators:
    :param labels:
    :param separator:
    :param tokens:
    :return:
    """
    all_features = []
    for feature_generator in feature_generators:
        features = feature_generator.generate_feature(tokens)
        all_features.append(features)

    featurized_output = []
    for i, token in enumerate(tokens):
        line = token
        for feature in all_features:
            line += '{}[{}]{}'.format(separator, feature[0], feature[1][i])
        if labels:
            line += separator + labels[i]
        else:
            line += separator + FAKE_LABEL
        featurized_output.append(line.strip() + '\n')

    return featurized_output
