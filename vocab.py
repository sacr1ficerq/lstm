from typing import Sequence, Iterable

from tqdm import tqdm

from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords
from nltk.stem import PorterStemmer

from gensim.corpora import Dictionary


class Vocab:
    def __init__(self, corpus: Sequence[str], no_below=1, no_above=0.8, max_tokens=100, clean=True):
        self.clean = clean
        self.max_tokens = max_tokens
        self.stemmer = PorterStemmer()

        tokenized_corpus = (self.tokenize(txt) for txt in tqdm(corpus))
        self.dictionary: Dictionary = Dictionary(tokenized_corpus)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)

        n = len(self.dictionary)
        self.special_tokens = {
            '<PAD>': n,
            '<UNK>': n + 1,
            '<BOS>': n + 2,
            '<EOS>': n + 3
        }

        self.pad_idx = self.special_tokens['<PAD>']

        for k, v in self.dictionary.token2id.items():
            self.dictionary.id2token[v] = k

        for k, v in self.special_tokens.items():
            self.dictionary.token2id[k] = v

    def tokenize(self, txt: str) -> Iterable[str]:
        if not self.clean:
            return txt.split(' ')
        CUSTOM_FILTERS = [
            strip_tags,
            strip_punctuation,
            strip_multiple_whitespaces,
            strip_numeric,
            remove_stopwords,
        ]
        res = preprocess_string(txt, CUSTOM_FILTERS)
        res = (self.stemmer.stem(word) for word in res)
        return res

    def txt2id(self, txt: str) -> Sequence[int]:
        tokens = self.tokenize(txt)
        res = self.dictionary.doc2idx(tokens, unknown_word_index=self.special_tokens['<UNK>'])
        res = list(filter(lambda x: x != self.special_tokens['<UNK>'], res))
        res = res[:self.max_tokens]
        return res

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
    print(example)
    vocab = Vocab([example])
    print(*vocab.tokenize(example), sep=' ')

