from typing import Sequence, Iterable
from tqdm import tqdm
import bpe_tokenizer as rnn_tokenizer
import re


class BPEVocab:
    def __init__(
        self, 
        corpus: Sequence[str], 
        vocab_size: int = 5000,
        max_tokens: int = 100, 
        clean: bool = True
    ):
        self.clean = clean
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size

        self.tokenizer = rnn_tokenizer.BPETokenizer(
            unk_token="<UNK>",
            pad_token="<PAD>"
        )

        processed = [self._preprocess(txt) for txt in tqdm(corpus, desc="Preprocessing")]
        self.tokenizer.build_vocab(processed, vocab_size=vocab_size - 2)  # -2 for BOS/EOS

        n = self.tokenizer.vocab_size()
        self.special_tokens = {
            '<PAD>': self.tokenizer.get_pad_id(),
            '<UNK>': self.tokenizer.get_unk_id(),
        }

        self.pad_idx = self.special_tokens['<PAD>']
        self.unk_idx = self.special_tokens['<UNK>']

    def _preprocess(self, txt: str) -> str:
        if not self.clean:
            return txt

        txt = txt.lower()
        txt = re.sub(r'<[^>]+>', '', txt)  # strip tags
        txt = re.sub(r'[^\w\s]', '', txt)  # strip punctuation
        txt = re.sub(r'\d+', '', txt)  # strip numeric
        txt = re.sub(r'\s+', ' ', txt)  # strip multiple whitespaces

        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'as', 'by', 'from'
        }
        words = txt.split()
        words = [w for w in words if w and w not in stopwords]

        return ' '.join(words)

    def tokenize(self, txt: str) -> Iterable[str]:
        processed = self._preprocess(txt)
        if not processed:
            return []

        # Encode and get back token strings
        token_ids = self.tokenizer.encode(processed)
        tokens = []
        for tid in token_ids:
            token = self.tokenizer.id_to_token(tid)
            if token:
                tokens.append(token)
        return tokens

    def txt2id(self, txt: str) -> Sequence[int]:
        processed = self._preprocess(txt)
        if not processed:
            return []

        ids = self.tokenizer.encode(processed)

        ids = [i for i in ids if i != self.unk_idx]

        ids = ids[:self.max_tokens]

        return ids

    def id2txt(self, ids: Sequence[int]) -> str:
        tokens = []
        for idx in ids:
            if idx == self.pad_idx:
                tokens.append('<PAD>')
            elif idx == self.unk_idx:
                tokens.append('<UNK>')
            else:
                token = self.tokenizer.id_to_token(idx)
                tokens.append(token if token else '<UNK>')
        return ' '.join(tokens)

    def save(self, path: str):
        self.tokenizer.save(path)

    def load(self, path: str):
        self.tokenizer.load(path)


if __name__ == '__main__':
    example = 'wasabi technologies the hot cloud storage company today announced a multi year deal to become the official cloud storage partner of the boston bruins'

    print(f"Original: {example}\n")

    vocab = BPEVocab([example], vocab_size=100)

    print(f"Tokens: {' '.join(vocab.tokenize(example))}\n")

    ids = vocab.txt2id(example)
    print(f"IDs: {ids}\n")

    decoded = vocab.id2txt(ids)
    print(f"Decoded: {decoded}\n")

    print(f"Vocab size: {vocab.tokenizer.vocab_size() + 2}")
    print(f"Special tokens: {vocab.special_tokens}")
