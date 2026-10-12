from __future__ import annotations

from collections import Counter
from typing import List, Tuple, Set, Union, Optional
from src.config import START_TOKEN, END_TOKEN, UNK_TOKEN

class NGramModel:
    def __init__(self, order: int = 3, min_word_frequency: int = 1):
        if order < 1:
            raise ValueError("A ordem do n-gram deve ser maior ou igual a 1.")
        self.order = order
        self.min_word_frequency = min_word_frequency
        
        self.vocab: Set[str] = set()
        self._unigram_counts: Counter[str] = Counter()
        self._ngram_counts: Counter[Tuple[str, ...]] = Counter()

    def fit(self, sentences: Union[List[List[str]], any]) -> NGramModel:
        if hasattr(sentences, "sentences"):
            sentence_list = sentences.sentences
        else:
            sentence_list = sentences

        for sent in sentence_list:
            # Vocabulário exclui delimitadores de borda (<s> e </s>)
            for w in sent:
                self.vocab.add(w)
                self._unigram_counts[w] += 1

            # N-grams com marcadores de borda para contagem
            padded_sent = [START_TOKEN] * (self.order - 1) + sent + [END_TOKEN]
            for i in range(len(padded_sent)):
                for k in range(1, self.order + 1):
                    if i + k <= len(padded_sent):
                        ngram = tuple(padded_sent[i:i + k])
                        self._ngram_counts[ngram] += 1
        return self

    def unigram_count(self, word: str) -> int:
        return self._unigram_counts.get(word, 0)

    def ngram_count(self, ngram: Tuple[str, ...]) -> int:
        return self._ngram_counts.get(tuple(ngram), 0)

    def map_tokens(self, tokens: List[str]) -> List[str]:
        return [t if t in self.vocab else UNK_TOKEN for t in tokens]

    @property
    def candidates(self) -> Set[str]:
        return self.vocab