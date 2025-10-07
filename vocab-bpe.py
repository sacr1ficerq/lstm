from collections import Counter, OrderedDict
from typing import Sequence, Iterable, List, Tuple, Dict
from tqdm import tqdm

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    remove_stopwords,
)
from nltk.stem import PorterStemmer


class Vocab:
    """
    Byte Pair Encoding (BPE) vocabulary builder.

    Parameters
    ----------
    corpus : Sequence[str]
        List of raw documents.
    vocab_size : int
        Maximum vocabulary size (including specials). Each merge adds one token.
    min_pair_freq : int
        Stop merging once the best pair appears fewer than this many times.
    lowercase : bool
        Apply lowercasing before the Gensim preprocessing filters.
    use_stemming : bool
        Optional Porter stemming after preprocessing.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        vocab_size: int = 1000,
        min_pair_freq: int = 2,
        lowercase: bool = True,
        use_stemming: bool = False,
    ):
        if vocab_size < 10:
            raise ValueError("vocab_size is too small for a useful BPE vocabulary.")

        self.vocab_size = vocab_size
        self.min_pair_freq = min_pair_freq
        self.lowercase = lowercase
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer()

        # Special symbols come first and get fixed ids.
        self.special_tokens = OrderedDict(
            [
                ("<PAD>", 0),
                ("<UNK>", 1),
                ("<BOS>", 2),
                ("<EOS>", 3),
            ]
        )
        self.pad_idx = self.special_tokens["<PAD>"]

        # Preprocess corpus (mirrors your original pipeline).
        processed_docs = [self._preprocess(doc) for doc in tqdm(corpus, desc="Preprocess")]

        # Word frequencies drive BPE statistics.
        self.word_freqs = self._build_word_freqs(processed_docs)
        if not self.word_freqs:
            raise ValueError("Corpus produced an empty vocabulary after preprocessing.")

        # Characters + end-of-word sentinel initialise the symbol set.
        self.symbols = self._collect_initial_symbols(self.word_freqs)
        merges_budget = max(
            0, self.vocab_size - len(self.special_tokens) - len(self.symbols)
        )

        # Learn merge rules.
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self._train_bpe(merges_budget)

        # Build lookup tables.
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_lookup_tables()

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def _preprocess(self, txt: str) -> List[str]:
        if self.lowercase:
            txt = txt.lower()
        CUSTOM_FILTERS = [
            strip_tags,
            strip_punctuation,
            strip_multiple_whitespaces,
            strip_numeric,
            remove_stopwords,
        ]
        tokens = preprocess_string(txt, CUSTOM_FILTERS)
        if self.use_stemming:
            tokens = [self.stemmer.stem(tok) for tok in tokens]
        return tokens

    @staticmethod
    def _build_word_freqs(docs: Sequence[List[str]]) -> Counter:
        freqs = Counter()
        for tokens in docs:
            freqs.update(tok for tok in tokens if tok)
        return freqs

    @staticmethod
    def _collect_initial_symbols(word_freqs: Counter) -> set:
        symbols = set()
        for word in word_freqs:
            symbols.update(word)
        symbols.add("</w>")
        return symbols

    @staticmethod
    def _get_pair_stats(vocab: Dict[Tuple[str, ...], int]) -> Counter:
        stats = Counter()
        for token_tuple, freq in vocab.items():
            for i in range(len(token_tuple) - 1):
                stats[(token_tuple[i], token_tuple[i + 1])] += freq
        return stats

    @staticmethod
    def _merge_vocab(
        pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        merged_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for token_tuple, freq in vocab.items():
            token_str = " ".join(token_tuple)
            merged_str = token_str.replace(bigram, replacement)
            merged_vocab[tuple(merged_str.split(" "))] = freq
        return merged_vocab

    def _train_bpe(self, merges_budget: int) -> None:
        vocab = {
            tuple(word) + ("</w>",): freq for word, freq in self.word_freqs.items()
        }
        for _ in range(merges_budget):
            stats = self._get_pair_stats(vocab)
            if not stats:
                break
            (a, b), freq = stats.most_common(1)[0]
            if freq < self.min_pair_freq:
                break
            vocab = self._merge_vocab((a, b), vocab)
            self.merges.append((a, b))
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

    # ------------------------------------------------------------------ #
    # Tokenisation
    # ------------------------------------------------------------------ #
    def _encode_word(self, word: str) -> List[str]:
        if not word:
            return []

        for ch in word:
            if ch not in self.symbols:
                return ["<UNK>"]

        symbols = list(word) + ["</w>"]
        while True:
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break
            ranked_pairs = [
                (self.merge_ranks.get(pair, float("inf")), pair) for pair in pairs
            ]
            best_rank, best_pair = min(ranked_pairs, key=lambda x: x[0])
            if best_rank == float("inf"):
                break

            first, second = best_pair
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        if symbols and symbols[-1] == "</w>":
            symbols = symbols[:-1]
        return symbols or ["<UNK>"]

    def tokenize(self, txt: str) -> List[str]:
        tokens = []
        for word in self._preprocess(txt):
            tokens.extend(self._encode_word(word))
        return tokens

    # ------------------------------------------------------------------ #
    # Lookup tables & conversions
    # ------------------------------------------------------------------ #
    def _build_lookup_tables(self) -> None:
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {idx: tok for tok, idx in self.special_tokens.items()}
        next_id = len(self.special_tokens)

        token_set = set()
        for word in self.word_freqs:
            token_set.update(self._encode_word(word))
        token_set.update(self.symbols)
        token_set.discard("</w>")  # sentinel only

        for token in sorted(token_set):
            if token in self.token_to_id:
                continue
            if next_id >= self.vocab_size:
                break
            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token
            next_id += 1

    def txt2id(
        self,
        txt: str,
        add_special_tokens: bool = False,
        drop_unk: bool = True,
    ) -> List[int]:
        tokens = self.tokenize(txt)
        ids = [self.token_to_id.get(tok, self.special_tokens["<UNK>"]) for tok in tokens]

        if drop_unk:
            ids = [idx for idx in ids if idx != self.special_tokens["<UNK>"]]

        if add_special_tokens:
            ids = (
                [self.special_tokens["<BOS>"]]
                + ids
                + [self.special_tokens["<EOS>"]]
            )
        return ids

    def id2txt(self, ids: Sequence[int]) -> str:
        words = []
        current_word = ""
        for idx in ids:
            token = self.id_to_token.get(idx, "<UNK>")
            if token in self.special_tokens:
                continue
            if token == "<UNK>":
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append("<UNK>")
                continue
            if token.endswith("</w>"):
                current_word += token[:-4]
                words.append(current_word)
                current_word = ""
            else:
                current_word += token
        if current_word:
            words.append(current_word)
        return " ".join(words)
