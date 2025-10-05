from typing import List, Tuple, Sequence
from torch.utils.data import DataLoader, Dataset
import torch

from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords
from nltk.stem import PorterStemmer

from gensim.corpora import Dictionary

class Vocab:
    def __init__(self, corpus: Sequence[str]):
        self.stemmer = PorterStemmer()

        tokenized_corpus = map(self.tokenize, corpus)
        self.dictionary: Dictionary = Dictionary(tokenized_corpus)
        self.dictionary.filter_extremes(no_below=1, no_above=0.8)

        n = len(self.dictionary)
        self.special_tokens = {
            '<UNK>': n,
            '<PAD>': n + 1,
            '<BOS>': n + 2,
            '<EOS>': n + 3
        }

        for k, v in self.dictionary.token2id.items():
            self.dictionary.id2token[v] = k

        for k, v in self.special_tokens.items():
            self.dictionary.token2id[k] = v

    def tokenize(self, txt: str) -> Sequence[str]:
        CUSTOM_FILTERS = [
            strip_tags,
            strip_punctuation,
            strip_multiple_whitespaces,
            strip_numeric,
            remove_stopwords,
            lambda text: ' '.join(map(self.stemmer.stem, text.split()))
        ]
        return preprocess_string(txt, CUSTOM_FILTERS)

    def txt2id(self, txt: str) -> Sequence[int]:
        tokens = self.tokenize(txt)
        res = self.dictionary.doc2idx(tokens, unknown_word_index=self.special_tokens['<UNK>'])
        return list(filter(lambda x: x != self.special_tokens['<UNK>'], res))

    def id2txt(self, ids: Sequence[int]) -> str:
        tokens = []
        for idx in ids:
            if idx in self.dictionary.id2token:
                tokens.append(self.dictionary.id2token[idx])
            else:
                tokens.append('<UNK>')
        return ' '.join(tokens)


if __name__ == '__main__':
    example = 'wasabi technologies the hot cloud storage company today announced a multi year deal to become the official cloud storage partner of the boston bruins'
    vocab = Vocab([example])
    print(*vocab.tokenize(example), sep='\n')

