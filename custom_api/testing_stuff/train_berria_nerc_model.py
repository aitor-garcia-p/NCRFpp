"""
This is a file to read/transform BERRIA NERC dataset and train a model using NCRFpp.
Let's see how far it gets...
"""
import os
import re
from typing import List

import es_core_news_sm


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


def generate_ncrfpp_dataset(input_folder, output):
    """
    Transforms the brat data to NCRFpp format (similar to CoNLL)
    :param input:
    :param output:
    :return:
    """
    files = os.listdir(input_folder)
    files_without_extension = [f for f in {re.sub('\.\w+$', '', file) for file in files}]
    files_without_extension.sort()

    output_lines = []

    for i, file in enumerate(files_without_extension):
        print('{}/{} Processing file {} ...'.format(i, len(files_without_extension), file))
        txt_file_name = file + '.txt'
        ann_file_name = file + '.ann'

        txt_content = read_file_content(input_folder, txt_file_name)
        ann_content = read_file_content(input_folder, ann_file_name)
        brat_annotations = parse_brat_ann_file(ann_content)

        doc = nlp(txt_content.strip())
        for sent in doc.sents:
            for token in doc[sent.start:sent.end]:
                if len(token.text.strip()) == 0:
                    # this is for the line breaks, spaCy seems to treat them as tokens, so we leverage them here to add a sentence separation
                    output_lines.append('')
                    continue
                offset = token.idx
                # print('Token offset {}, (now plus sent) {}'.format(offset, sent.start + offset))
                # print(token, offset)
                tag = return_tag(brat_annotations, offset)
                output_lines.append('{} {}'.format(token.text, tag))
            output_lines.append('')

    return output_lines
    # print(files_without_extension)


def read_file_content(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
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


def return_tag(brat_annotations: List[BratAnnotation], token_offset):
    """
    BIO tags, hardcoded (I am doing this fast, ok?)
    :param brat_annotations:
    :param token_offset:
    :return:
    """
    for brat_annotation in brat_annotations:
        if token_offset == brat_annotation.start:
            return 'B-{}'.format(brat_annotation.ent_type)
        elif brat_annotation.start < token_offset < brat_annotation.end:
            return 'I-{}'.format(brat_annotation.ent_type)
        # elif token_offset < brat_annotation.end:
        #     # annotations are in order, token is not represented
        #     # print(brat_annotation.start, brat_annotation.end, token_offset)
        #     return 'O'
    return 'O'


if __name__ == '__main__':
    input_folder = 'C:\\Users\\agarciap\\Data\\DATASETS\\NER_annotation_BERRIA_dataset_Montse'
    output = None
    output_lines = generate_ncrfpp_dataset(input_folder, output)
    for line in output_lines[:100]:
        print(line)
