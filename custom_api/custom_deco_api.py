"""
Remarks about this API:
 - the idea is to provide text input and retrieve it labelled
 - the model is assumed to be already loaded
 - the feature-generation to represent the input before querying the model is responsibility of the tool
 - the input is expected to be a single piece of raw of text, tokenization is not responsibility of the client
 - it is responsibility of the tool to use the same tokenisation (or the most similar one) than the one used for training
"""
import os
import sys

import time

import es_core_news_sm
from typing import List, Tuple, Dict
import torch

# import spacy

from custom_api.feature_generators import FeatureGenerator, AffixFeatureGenerator, SpacyBasedFeatureGenerator, \
    DictionaryBasedFeatureGenerator, \
    CapitalizationFeatureGenerator, WordPositionFeatureGenerator
# from my_main2 import label_input
# this is the stuff imported from the toolkit
from model.sentclassifier import SentClassifier
from model.seqlabel import SeqLabel
from utils.data import Data
from utils.functions import normalize_word
from main import recover_nbest_label, get_ner_fmeasure, recover_label, batchify_with_label

FEATURES = []


class CustomDecodingAPI:

    @staticmethod
    def default_tokenizer(input_text: str) -> List[str]:
        # naive, just to test
        return input_text.split(' ')

    def __init__(self, feature_generators: List[FeatureGenerator], tokenizer=default_tokenizer):
        self.feature_generators = feature_generators  # I am not sure what this should look like...
        self.tokenizer = tokenizer

    def deco_input(self, input_text: str):
        model_dir = 'C:\\Users\\agarciap\\Data\\DATASETS\\NCRFpp_tests\\test_with_crf'
        model_name = 'lstmcrf'

        tokens = self.tokenizer(input_text)  # self.tokenize_input(input_text)
        input_lines = self.compute_features(tokens, separator=' ', fake_label='O')

        decode_results, pred_scores = self.label_input(input_lines, model_dir, model_name, nbest=5)
        return decode_results, pred_scores

    def compute_features(self, tokens: List[str], separator: str = '\t', fake_label: str = None) \
            -> List[Tuple[str, Dict[str, str]]]:
        """
        Receives a list of tokens, and computes the features for each of them (where are those features specified?)
        The result is a list of tuples, containing each token and a dictionary of features
        :param fake_label:
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

    def label_input(self, input_lines, model_dir, model_name, nbest=None):
        data = Data()
        data.HP_gpu = torch.cuda.is_available()

        (model_dset, model_file) = obtain_model_dset_and_file(model_dir, model_name)
        data.load(model_dset)
        data.dset_dir = model_dset
        data.load_model_dir = model_file

        # data.use_crf=False

        data.HP_gpu = False

        # data.show_data_summary()
        status = data.status.lower()
        data.fix_alphabet()
        data.nbest = nbest

        print("LABEL ALPHABET SIZE", len(data.label_alphabet.instance2index))
        print("LABEL ALPHABET", data.label_alphabet.instance2index)

        data.raw_texts, data.raw_Ids = self.read_instance(input_lines, data.word_alphabet, data.char_alphabet,
                                                          data.feature_alphabets, data.label_alphabet,
                                                          data.number_normalized, data.MAX_SENTENCE_LENGTH,
                                                          data.sentence_classification, data.split_token)

        decode_results, pred_scores = self.load_model_decode(data, 'raw')

        print("LABEL ALPHABET SIZE", len(data.label_alphabet.instance2index))
        print("LABEL ALPHABET", data.label_alphabet.instance2index)
        for i in range(12):
            print(i, data.label_alphabet.get_instance(i))

        return decode_results, pred_scores

    def load_model_decode(self, data, name):
        print("Load Model from file: ", data.model_dir)
        if data.sentence_classification:
            model = SentClassifier(data)
        else:
            model = SeqLabel(data)
        # model = SeqModel(data)
        # load model need consider if the model trained in GPU and load in CPU, or vice versa
        # if not gpu:
        #     model.load_state_dict(torch.load(model_dir))
        #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
        #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
        # else:
        #     model.load_state_dict(torch.load(model_dir))
        #     # model = torch.load(model_dir)
        model.load_state_dict(torch.load(data.load_model_dir, map_location='cpu'))

        print("Decode %s data, nbest: %s ..." % (name, data.nbest))
        start_time = time.time()
        speed, acc, p, r, f, pred_results, pred_scores = self.evaluate(data, model, name, data.nbest)
        end_time = time.time()
        time_cost = end_time - start_time
        if data.seg:
            print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                name, time_cost, speed, acc, p, r, f))
        else:
            print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
        return pred_results, pred_scores

    def evaluate(self, data, model, name, nbest=None):
        instances = data.raw_Ids

        right_token = 0
        whole_token = 0
        nbest_pred_results = []
        pred_scores = []
        pred_results = []
        gold_results = []
        ## set model in eval model
        model.eval()
        batch_size = data.HP_batch_size
        start_time = time.time()
        train_num = len(instances)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = instances[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, data.HP_gpu, False, data.sentence_classification)
            if nbest and not data.sentence_classification:
                scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
                                                           batch_charlen, batch_charrecover, mask,
                                                           nbest)
                nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
                nbest_pred_results += nbest_pred_result
                pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
                # select the best sequence to evaluate
                tag_seq = nbest_tag_seq[:, :, 0]
            else:
                tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                                mask)
            # print("tag:",tag_seq)
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover,
                                                   data.sentence_classification)
            pred_results += pred_label
            gold_results += gold_label
        decode_time = time.time() - start_time
        speed = len(instances) / decode_time if decode_time > 0 else float('inf')
        acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
        if nbest and not data.sentence_classification:
            return speed, acc, p, r, f, nbest_pred_results, pred_scores
        return speed, acc, p, r, f, pred_results, pred_scores

    def read_instance(self, input_lines, word_alphabet, char_alphabet, feature_alphabets, label_alphabet,
                      number_normalized,
                      max_sent_length,
                      sentence_classification=False, split_token='\\t', char_padding_size=-1,
                      char_padding_symbol='</pad>'):
        feature_num = len(feature_alphabets)
        in_lines = input_lines  # open(input_file,'r', encoding="utf8").readlines()
        instence_texts = []
        instence_Ids = []
        words = []
        features = []
        chars = []
        labels = []
        word_Ids = []
        feature_Ids = []
        char_Ids = []
        label_Ids = []

        # if sentence classification data format, splited by \\t
        if sentence_classification:
            for line in in_lines:
                if len(line) > 2:
                    pairs = line.strip().split(split_token)
                    sent = pairs[0]
                    if sys.version_info[0] < 3:
                        sent = sent.decode('utf-8')
                    original_words = sent.split()
                    for word in original_words:
                        words.append(word)
                        if number_normalized:
                            word = normalize_word(word)
                        word_Ids.append(word_alphabet.get_index(word))
                        # get char
                        char_list = []
                        char_Id = []
                        for char in word:
                            char_list.append(char)
                        if char_padding_size > 0:
                            char_number = len(char_list)
                            if char_number < char_padding_size:
                                char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                            assert (len(char_list) == char_padding_size)
                        for char in char_list:
                            char_Id.append(char_alphabet.get_index(char))
                        chars.append(char_list)
                        char_Ids.append(char_Id)

                    label = pairs[-1]
                    label_Id = label_alphabet.get_index(label)
                    # get features
                    feat_list = []
                    feat_Id = []
                    for idx in range(feature_num):
                        feat_idx = pairs[idx + 1].split(']', 1)[-1]
                        feat_list.append(feat_idx)
                        feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                    # combine together and return, notice the feature/label as different format with sequence labeling task
                    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                        instence_texts.append([words, feat_list, chars, label])
                        instence_Ids.append([word_Ids, feat_Id, char_Ids, label_Id])
                    words = []
                    features = []
                    chars = []
                    char_Ids = []
                    word_Ids = []
                    feature_Ids = []
                    label_Ids = []
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                instence_texts.append([words, feat_list, chars, label])
                instence_Ids.append([word_Ids, feat_Id, char_Ids, label_Id])
                words = []
                features = []
                chars = []
                char_Ids = []
                word_Ids = []
                feature_Ids = []
                label_Ids = []

        else:
            ### for sequence labeling data format i.e. CoNLL 2003
            for line in in_lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if sys.version_info[0] < 3:
                        word = word.decode('utf-8')
                    words.append(word)
                    if number_normalized:
                        word = normalize_word(word)
                    label = pairs[-1]
                    labels.append(label)
                    word_Ids.append(word_alphabet.get_index(word))
                    label_Ids.append(label_alphabet.get_index(label))
                    ## get features
                    feat_list = []
                    feat_Id = []
                    for idx in range(feature_num):
                        feat_idx = pairs[idx + 1].split(']', 1)[-1]
                        feat_list.append(feat_idx)
                        feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                    features.append(feat_list)
                    feature_Ids.append(feat_Id)
                    ## get char
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                        assert (len(char_list) == char_padding_size)
                    else:
                        ### not padding
                        pass
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)
                else:
                    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                        instence_texts.append([words, features, chars, labels])
                        instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids])
                    words = []
                    features = []
                    chars = []
                    labels = []
                    word_Ids = []
                    feature_Ids = []
                    char_Ids = []
                    label_Ids = []
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                instence_texts.append([words, features, chars, labels])
                instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids])
                words = []
                features = []
                chars = []
                labels = []
                word_Ids = []
                feature_Ids = []
                char_Ids = []
                label_Ids = []
        return instence_texts, instence_Ids


def obtain_model_dset_and_file(model_dir, model_name):
    model_dset = os.path.join(model_dir, model_name + '.dset')
    model_file = os.path.join(model_dir, model_name + '.model')
    return model_dset, model_file


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

    brown4 = DictionaryBasedFeatureGenerator(feature_name='BROWN4', input_path=PATH_TO_CLUSTERS_2, unknown_value='0',
                                             trim_value_at=4)
    brown6 = DictionaryBasedFeatureGenerator(feature_name='BROWN6', input_path=PATH_TO_CLUSTERS_2, unknown_value='0',
                                             trim_value_at=6)
    brown10 = DictionaryBasedFeatureGenerator(feature_name='BROWN10', input_path=PATH_TO_CLUSTERS_2, unknown_value='0',
                                              trim_value_at=10)
    brown20 = DictionaryBasedFeatureGenerator(feature_name='BROWN20', input_path=PATH_TO_CLUSTERS_2, unknown_value='0',
                                              trim_value_at=20)

    c1 = DictionaryBasedFeatureGenerator(feature_name='C1', input_path=PATH_TO_CLUSTERS_1, unknown_value='0')
    c3 = DictionaryBasedFeatureGenerator(feature_name='C3', input_path=PATH_TO_CLUSTERS_3, unknown_value='0')
    c4 = DictionaryBasedFeatureGenerator(feature_name='C4', input_path=PATH_TO_CLUSTERS_4, unknown_value='0')
    c5 = DictionaryBasedFeatureGenerator(feature_name='C5', input_path=PATH_TO_CLUSTERS_5, unknown_value='0')

    nlp = es_core_news_sm.load()
    spacy_pos = SpacyBasedFeatureGenerator(feature_name='POS', spacy_model=nlp, feature_type='pos')
    spacy_lemma = SpacyBasedFeatureGenerator(feature_name='LEM', spacy_model=nlp, feature_type='lemma')

    custom_deco_api = CustomDecodingAPI(
        feature_generators=[spacy_pos, cap, is_first, allcaps, somecaps, spacy_lemma, prefix3, prefix4, suffix1,
                            suffix2, suffix3, suffix4, brown4,
                            brown6, brown10, brown20, c1, c3, c4, c5])
    generated_features = custom_deco_api.compute_features(tokens='esto es una prueba con palabras bonitas'.split(' '),
                                                          separator=' ')

    [print(output) for output in generated_features]

    generated_features = custom_deco_api.compute_features(tokens='gaur Donostian denbora ona egiten du'.split(' '),
                                                          separator=' ')
    [print(output) for output in generated_features]

    deco, probs = custom_deco_api.deco_input('gaur Bilbon denbora ona egiten du eta Pepe Goterak hitz egingo du')
    print(deco)
    print(probs)
