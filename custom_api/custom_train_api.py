import os
import tempfile
from typing import List

# I have copied the whole 'main' into a package to ease the packetization
import custom_api.copied_main as main
from custom_api.feature_generators import FeatureGenerator
from sklearn.model_selection import train_test_split

from custom_api.helpers import compute_features
from utils.data import Data


# TEST_MSG = 'this is supposed to be the custom train API, congrats...'


class CustomTrainApi:

    def __init__(self, feature_generators: List[FeatureGenerator], config_file_path: str):
        """
        Note that the features are somewhat fixed, also represented in the config file.
        This is not (at least for now) something to let the user to play with.
        :param feature_generators:
        :param config_file_path:
        """
        self.config_file_path = config_file_path
        self.feature_generators = feature_generators

    def prepare_data_and_train(self, train_set: str, output_model_dir: str, output_model_name: str, seed: int = 1234):
        """
        The hyper-parameters are supposed to be fixed in the config file.
        We will read them from there and make use of the very logic to let the toolkit make its work.
        The only special thing here is to split provided train_set into train/dev/test and use feature generators to... generate features
        :return:
        """
        # create output directory if not exist
        if not os.path.exists(output_model_dir):
            os.makedirs(output_model_dir)

        sentences = split_train_in_sentences(train_set)

        train_dev, test = train_test_split(sentences, test_size=0.2, random_state=seed)

        train, dev = train_test_split(train_dev, test_size=0.2, random_state=seed)

        # print(train[:10])

        train, dev, test = self.featurize(train), self.featurize(dev), self.featurize(test)

        _, train_path = tempfile.mkstemp()
        _, dev_path = tempfile.mkstemp()
        _, test_path = tempfile.mkstemp()

        print('Train temp path:', train_path)
        print('Dev temp path:', dev_path)
        print('Test temp path', test_path)

        # here the paths still point to empty files, will be filled next
        data = read_config_file(self.config_file_path)
        override_configuration(data, train_path, dev_path, test_path, os.path.join(output_model_dir, output_model_name))

        try:
            with open(train_path, 'w', encoding='utf-8') as train_file, \
                    open(dev_path, 'w', encoding='utf-8') as dev_file, open(test_path, 'w', encoding='utf-8') as test_file:
                train_file.write(train)
                dev_file.write(dev)
                test_file.write(test)

            with open(train_path, 'r', encoding='utf-8') as train_file:
                content = train_file.read()
                print('Checking what is inside train file...', content[:200])

            data_initialization(data)
            data.generate_instance('train')
            data.generate_instance('dev')
            data.generate_instance('test')
            data.build_pretrain_emb()
            main.train(data)
        except Exception as e:
            print('Exception: {}'.format(str(e)))
            raise e

        finally:
            os.remove(train_path)
            os.remove(dev_path)
            os.remove(test_path)

    def featurize(self, dataset):
        featurized_sentences = []
        # we expect only token<space>label per line (i.e. per list item)
        for sentence in dataset:
            # print(sentence)
            tokens, labels = zip(*[(x.strip().split(' ')[0], x.strip().split(' ')[1]) for x in sentence if len(sentence) > 0])
            # this is a list of strings containing a token with its features and label
            sentence_with_features = compute_features(tokens, self.feature_generators, separator=' ', labels=labels)
            as_str = ''.join(sentence_with_features).strip()
            featurized_sentences.append(as_str)
        # join individual sentences with a blank line
        return '\n\n'.join(featurized_sentences).strip()


def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def override_configuration(data, train_path, dev_path, test_path, output_path):
    """
    Given an already filled Data object, override some parts, like the train/dev/test, or the output path
    :param dev_path:
    :param test_path:
    :param output_path:
    :param train_path:
    :param data:
    :return:
    """
    data.train_dir = train_path
    data.dev_dir = dev_path
    data.test_dir = test_path
    data.model_dir = output_path


def read_config_file(path):
    """
    Read the config file, just like the original tool does
    :param path:
    :return:
    """
    data = Data()
    data.read_config(path)
    return data


def split_train_in_sentences(lines: List[str]) -> List[List[str]]:
    """
    Receives the data in the expected format (one token per line, empty line to separate sentences) and returns a list of sentences (lists of lines)
    :param lines:
    :return:
    """
    sentences = []
    current_sentence = []
    for line in lines:
        if line.strip() == '' and len(current_sentence) > 0:
            sentences.append(current_sentence)
            current_sentence = []
        else:
            current_sentence.append(line)
    # last sentence in case the file ends with no empty line
    if len(current_sentence) > 0:
        sentences.append(current_sentence)
    return sentences


if __name__ == '__main__':
    custom_train_api = CustomTrainApi([], '../my_demo.train.config')
    dataset_path = 'C:\\Users\\agarciap\\Dropbox\\datasets\\egunkaria\\named_ent_eu.test'
    with open(dataset_path, 'r', encoding='utf-8') as f:
        content_lines = f.readlines()
    print(content_lines[:50])
    custom_train_api.prepare_data_and_train(content_lines, output_model_dir='output_test_dir', output_model_name='super_test_model')
