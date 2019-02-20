"""
The idea is to put here, in homogeneous methods, the generation of each desired feature.
Of course it would be better to externalize each generator extending from a base generator.
But it will probably too much overhead for a low gain. It could be a future refactoring if it becomes necessary.
Thanks to the fact that in python the functions can be treated as objects this should suffice...
"""
from typing import List, Tuple


def generate_feat1(tokens: List[str]) -> Tuple[str, List[str]]:
    """
    As an example...
    It should return a tuple of "feature name" and the feature values for each token, with a 1:1 correspondence
    :param tokens: for example [tok1, tok2, tok3]
    :return: for example (feat1, [tok1_feat, tok2_feat, tok3_feat])
    """
    pass


def generate_suffix2(tokens: List[str]) -> Tuple[str, List[str]]:
    n = 2
    return generate_suffix_n(tokens, n)


def generate_suffix3(tokens: List[str]) -> Tuple[str, List[str]]:
    n = 3
    return generate_suffix_n(tokens, n)




def generate_suffix_n(tokens: List[str], n: int) -> Tuple[str, List[str]]:
    """
    As an example...
    It should return a tuple of "feature name" and the feature values for each token, with a 1:1 correspondence
    :param n: the number of characters for the suffix
    :param tokens: for example [tok1, tok2, tok3]
    :return: for example (feat1, [tok1_feat, tok2_feat, tok3_feat])
    """
    feature_values = [tok[-n:] if len(tok) > n else tok for tok in tokens]
    return 'suffix' + str(n), feature_values


def generate_preffix_n(tokens: List[str], n: int) -> Tuple[str, List[str]]:
    """
    As an example...
    It should return a tuple of "feature name" and the feature values for each token, with a 1:1 correspondence
    :param n: the number of characters for the prefix
    :param tokens: for example [tok1, tok2, tok3]
    :return: for example (feat1, [tok1_feat, tok2_feat, tok3_feat])
    """
    feature_values = [tok[:n] if len(tok) > n else tok for tok in tokens]
    return 'prefix' + str(n), feature_values


if __name__ == '__main__':
    feature_gens = [generate_suffix2, generate_suffix3]

    tokens = 'this is a test'.split(' ')
    for feature_gen in feature_gens:
        print(feature_gen(tokens))
    # print(generate_suffix2('this is a test'.split(' ')))

    for i in range(1,3):
        print(generate_preffix_n(tokens, i))
