"""
Remarks about this API:
 - the idea is to provide text input and retrieve it labelled
 - the model is assumed to be already loaded
 - the feature-generation to represent the input before querying the model is responsibility of the tool
 - the input is expected to be a single piece of raw of text, tokenization is not responsibility of the client
 - it is responsibility of the tool to use the same tokenisation (or the most similar one) than the one used for training
"""
from typing import List, Tuple, Dict

FEATURES = []


class CustomDecodingAPI:

    def __init__(self):
        self.features = {}  # I am not sure what this should look like...
        pass

    @staticmethod
    def deco_input(input_text: str):
        pass

    @staticmethod
    def tokenize_input(input_text: str) -> List[str]:
        pass

    def compute_features(self, tokens: List[str]) -> List[Tuple[str, Dict[str:str]]]:
        """
        Receives a list of tokens, and computes the features for each of them (where are those features specified?)
        The result is a list of tuples, containing each token and a dictionary of features
        :param tokens:
        :return:
        """
        features = self.features
        # not really necessary...
        # some features might need the whole text to be computed (e.g. part-of-speech)
        # so better that the feature generators admit full texts (sentences or whatever)
        # and internally assign the feature individually or in group

        pass
