"""
This is a file to read/transform BERRIA NERC dataset and train a model using NCRFpp.
Let's see how far it gets...
"""
import os
import re
from typing import List, Dict

import es_core_news_sm
from sklearn.model_selection import train_test_split


class BratAnnotation:

    def __init__(self, ent_type, start: int, end: int, text):
        self.end = end
        self.start = start
        self.ent_type = ent_type
        self.text = text


# load for tokenisation, it is Spanish, but only for tokenisation should be enough
nlp = es_core_news_sm.load(disable=["parser", "tagger", "ner"])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def load_token_tags_from_brat_files(input_folder, entity_type_mapping: Dict[str, str]):
    """
    Transforms the brat data to NCRFpp format (similar to CoNLL)
    :return:
    """
    entity_type_mapping = entity_type_mapping or {}
    files = os.listdir(input_folder)
    files_without_extension = [f for f in {re.sub('\.\w+$', '', file) for file in files}]
    files_without_extension.sort()

    output_lines = []

    for i, file in enumerate(files_without_extension):
        if file == '':
            # this happens for .stats_cache file
            continue
        print('{}/{} Processing file {} ...'.format(i, len(files_without_extension), file))
        txt_file_name = file + '.txt'
        ann_file_name = file + '.ann'

        txt_content = read_file_content(input_folder, txt_file_name, newline='\r\n')
        ann_content = read_file_content(input_folder, ann_file_name, newline='\r\n')
        brat_annotations = parse_brat_ann_file(ann_content)

        doc = nlp(txt_content.strip())

        last_annotation_used = None

        for sent in doc.sents:
            for token in doc[sent.start:sent.end]:
                if len(token.text.strip()) == 0:
                    # this is for the line breaks, spaCy seems to treat them as tokens, so we leverage them here to add a sentence separation
                    output_lines.append('')
                    continue
                offset = token.idx
                # print('Token offset {}, (now plus sent) {}'.format(offset, sent.start + offset))
                # print(token, offset)
                tag, last_annotation_used = return_tag(brat_annotations, offset, last_annotation_used)
                tag = reprocess_tag(tag, entity_type_mapping)
                output_lines.append('{} {}'.format(token.text, tag))
            # to prevent adding a double line break
            if len(output_lines) > 0 and output_lines[-1].strip() != '':
                output_lines.append('')

    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.writelines([line + '\n' for line in output_lines])
    return output_lines
    # print(files_without_extension)


def reprocess_tag(tag, entity_type_mapping):
    if '-' in tag:
        bio_mark, entity_type = tuple(tag.split('-'))
        if entity_type in entity_type_mapping:
            if entity_type_mapping[entity_type] is None:
                print('Ignoring tag {}, mapping to O'.format(tag))
                return 'O'
            mapped_tag = bio_mark + '-' + entity_type_mapping[entity_type]
            print('Mapping tag {} to {}'.format(tag, mapped_tag))
            return mapped_tag
    return tag


def read_file_content(folder_path, file_name, newline='\r\n'):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8', newline=newline) as f:
        content = f.read()
    return content


def parse_brat_ann_file(ann_file_content) -> List[BratAnnotation]:
    brat_annotations = []
    lines = ann_file_content.split('\n')
    for line in lines:
        if len(line.strip()) == 0:
            continue
        # print(line.split('\t'))
        _, ent_type_and_offset, text = tuple(line.split('\t'))
        ent_type, start, end = ent_type_and_offset.split(' ')
        brat_annotation = BratAnnotation(ent_type, int(start), int(end), text)
        brat_annotations.append(brat_annotation)
    return brat_annotations


# def return_tag(brat_annotations: List[BratAnnotation], token_offset):
#     """
#     BIO tags, hardcoded (I am doing this fast, ok?)
#     :param brat_annotations:
#     :param token_offset:
#     :return:
#     """
#     for brat_annotation in brat_annotations:
#         if token_offset == brat_annotation.start:
#             return 'B-{}'.format(brat_annotation.ent_type)
#         elif brat_annotation.start < token_offset < brat_annotation.end:
#             return 'I-{}'.format(brat_annotation.ent_type)
#         # elif token_offset < brat_annotation.end:
#         #     # annotations are in order, token is not represented
#         #     # print(brat_annotation.start, brat_annotation.end, token_offset)
#         #     return 'O'
#     return 'O'

def return_tag(brat_annotations: List[BratAnnotation], token_offset, last_annotation_used):
    """
    BIO tags, hardcoded (I am doing this fast, ok?)
    :param brat_annotations:
    :param token_offset:
    :return:
    """
    tag_to_return = 'O'
    for brat_annotation in brat_annotations:
        if token_offset == brat_annotation.start:
            # print('brat-ann start:{}, end:{}, token-offset:{}, returning B-tag'.format(brat_annotation.start, brat_annotation.end, token_offset))
            tag_to_return = 'B-{}'.format(brat_annotation.ent_type)
            last_annotation_used = brat_annotation
            break
        elif brat_annotation.start < token_offset < brat_annotation.end:
            # print('Brat annotation: {}'.format(brat_annotation.text))
            # print('brat-ann start:{}, end:{}, token-offset:{}, returning I-tag'.format(brat_annotation.start, brat_annotation.end, token_offset))
            if (last_annotation_used and last_annotation_used != brat_annotation) or last_annotation_used is None:
                # this is a double-check, this annotation has NOT been used yet, so regardless of the (misleading) offsets it deserves a B-tag
                tag_to_return = 'B-{}'.format(brat_annotation.ent_type)
            else:
                tag_to_return = 'I-{}'.format(brat_annotation.ent_type)
            last_annotation_used = brat_annotation
            break
        # elif token_offset < brat_annotation.end:
        #     # annotations are in order, token is not represented
        #     # print(brat_annotation.start, brat_annotation.end, token_offset)
        #     return 'O'

    return tag_to_return, last_annotation_used


def dump_for_ncrfpp(output_lines, output_files):
    train, test = train_test_split(output_lines, test_size=0.2, random_state=42)

    with open(output_files[0], 'w', encoding='utf-8') as f:
        f.writelines([line + '\n' for line in train])
    with open(output_files[1], 'w', encoding='utf-8') as f:
        f.writelines([line + '\n' for line in test])


if __name__ == '__main__':
    # input_folder = 'C:\\Users\\agarciap\\Data\\DATASETS\\NER_annotation_BERRIA_dataset_Montse'
    # output = 'C:\\Users\\agarciap\\Data\\DATASETS\\NCRFpp_tests\\BERRIA_dataset_Montse_NCRFpp_PerLocOrgMisc.txt'
    # generate_ncrfpp_dataset(input_folder, output,
    #                         entity_type_mapping={'GSP': 'Locations', 'Facilities': 'Locations', 'Products': 'Organization', 'Time': 'MISC'})

    input_folder = 'C:\\Users\\agarciap\\Data\\GAMES_DATA\\NERC_ANNOTATED_DATA\\tokikom_annotated_(zuriñe)'
    output_path = 'C:\\Users\\agarciap\\Data\\GAMES_DATA\\NERC_ANNOTATED_DATA\\tokikom_annotated_(zuriñe)'

    token_tags = []

    input_subfolders = os.listdir(input_folder)
    for subfolder in input_subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        print('Going to process subfolder {}'.format(subfolder_path))
        token_tags_for_this_folder = load_token_tags_from_brat_files(subfolder_path,
                                                                     entity_type_mapping={'GSP': 'Locations', 'Facilities': 'Locations',
                                                                                          'Products': 'Organization',
                                                                                          'Time': 'MISC'})
        token_tags += token_tags_for_this_folder

    dump_for_ncrfpp(token_tags, (output_path + '_train.ncrfpp.txt', output_path + '_test.ncrfpp.txt'))
