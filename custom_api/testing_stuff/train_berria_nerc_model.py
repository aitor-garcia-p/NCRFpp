"""
This is a file to read/transform BERRIA NERC dataset and train a model using NCRFpp.
Let's see how far it gets...
"""
import spacy

from custom_api.custom_train_api import CustomTrainApi
from custom_api.feature_generators import DictionaryBasedFeatureGenerator, AffixFeatureGenerator, CapitalizationFeatureGenerator, \
    WordPositionFeatureGenerator, SpacyBasedFeatureGenerator

if __name__ == '__main__':
    # input_folder = 'C:\\Users\\agarciap\\Data\\DATASETS\\NER_annotation_BERRIA_dataset_Montse'
    # output = 'C:\\Users\\agarciap\\Data\\DATASETS\\NCRFpp_tests\\BERRIA_dataset_Montse_NCRFpp.txt'
    # generate_ncrfpp_dataset(input_folder, output)

    # TRAINING STUFF

    print('Calculating features... may take some time depending on which features are enabled...')
    ####
    PATH_TO_CLUSTERS_1 = "/DATA/agarciap_data/GAMES_GN_stuff/nerc_training_related/eu_clusters_files/berrias50w5600"
    PATH_TO_CLUSTERS_2 = "/DATA/agarciap_data/GAMES_GN_stuff/nerc_training_related/eu_clusters_files/egunkariaprecleantokc1000p1txt"
    PATH_TO_CLUSTERS_3 = "/DATA/agarciap_data/GAMES_GN_stuff/nerc_training_related/eu_clusters_files/egunkarias50w5300"
    PATH_TO_CLUSTERS_4 = "/DATA/agarciap_data/GAMES_GN_stuff/nerc_training_related/eu_clusters_files/egunkariatokpunctlower200"
    PATH_TO_CLUSTERS_5 = "/DATA/agarciap_data/GAMES_GN_stuff/nerc_training_related/eu_clusters_files/euwikitokpunctlower200bak"
    # PATH_TO_CLUSTERS_1 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\berrias50w5600"
    # PATH_TO_CLUSTERS_2 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\egunkariaprecleantokc1000p1txt"
    # PATH_TO_CLUSTERS_3 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\egunkarias50w5300"
    # PATH_TO_CLUSTERS_4 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\egunkariatokpunctlower200"
    # PATH_TO_CLUSTERS_5 = "C:\\Users\\agarciap\\Data\\DATASETS\\basque_clusters_from_ixapipesnerc\\euwikitokpunctlower200bak"
    ####

    brown4 = DictionaryBasedFeatureGenerator(feature_name='BROWN4', input_path=PATH_TO_CLUSTERS_2, unknown_value='0',
                                             values_separator='\t',
                                             trim_value_at=4)
    brown6 = DictionaryBasedFeatureGenerator(feature_name='BROWN6', input_path=PATH_TO_CLUSTERS_2, unknown_value='0',
                                             values_separator='\t',
                                             trim_value_at=6)
    brown10 = DictionaryBasedFeatureGenerator(feature_name='BROWN10', input_path=PATH_TO_CLUSTERS_2, unknown_value='0',
                                              values_separator='\t',
                                              trim_value_at=10)
    brown20 = DictionaryBasedFeatureGenerator(feature_name='BROWN20', input_path=PATH_TO_CLUSTERS_2, unknown_value='0',
                                              values_separator='\t',
                                              trim_value_at=20)

    c1 = DictionaryBasedFeatureGenerator(feature_name='C1', input_path=PATH_TO_CLUSTERS_1, unknown_value='0')
    c3 = DictionaryBasedFeatureGenerator(feature_name='C3', input_path=PATH_TO_CLUSTERS_3, unknown_value='0')
    c4 = DictionaryBasedFeatureGenerator(feature_name='C4', input_path=PATH_TO_CLUSTERS_4, unknown_value='0')
    c5 = DictionaryBasedFeatureGenerator(feature_name='C5', input_path=PATH_TO_CLUSTERS_5, unknown_value='0')

    prefix3 = AffixFeatureGenerator('prefix3', 'prefix', 3)
    prefix4 = AffixFeatureGenerator('prefix4', 'prefix', 4)

    suffix2 = AffixFeatureGenerator('suffix2', 'suffix', 2)
    suffix3 = AffixFeatureGenerator('suffix3', 'suffix', 3)
    suffix4 = AffixFeatureGenerator('suffix4', 'suffix', 4)

    first_cap = CapitalizationFeatureGenerator('CAP', caps_type='first')
    all_caps = CapitalizationFeatureGenerator('ALLCAPS', caps_type='all')
    some_caps = CapitalizationFeatureGenerator('SOMECAPS', caps_type='some')

    is_first = WordPositionFeatureGenerator('FIRST', position_type='first')

    #############
    nlp = spacy.load('/DATA/agarciap_data/GAMES_GN_stuff/nerc_training_related/spacy_models_for_pos/model-final')
    # nlp = spacy.load('C:\\Users\\agarciap\\Data\\INTERCHANGE\\model-final')
    pos = SpacyBasedFeatureGenerator('POS', nlp, 'pos')
    #############

    custom_train_api = CustomTrainApi(
        [first_cap, all_caps, some_caps, is_first, brown4, brown6, brown10, brown20, c1, c3, c4, c5, prefix3, prefix4, suffix2, suffix3, suffix4],
        '../../my_demo.train.config')
    # dataset_path = 'C:\\Users\\agarciap\\Data\\DATASETS\\NCRFpp_tests\\BERRIA_dataset_Montse_NCRFpp_PerLocOrgMisc.txt'
    dataset_path = '/DATA/agarciap_data/GAMES_GN_stuff/nerc_training_related/BERRIA_dataset_Montse_NCRFpp_PerLocOrgMisc.txt'
    # dataset_path = "C:\\Users\\agarciap\\repositories\\V2_REPOS\\NCRFpp_forked\\sample_data\\train.bmes"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        content_lines = f.readlines()
    print(content_lines[:50])
    custom_train_api.prepare_data_and_train(content_lines, output_model_dir='output_test_dir_BERRIA', output_model_name='super_test_model_BERRIA')
