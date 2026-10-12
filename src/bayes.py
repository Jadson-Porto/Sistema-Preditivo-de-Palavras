from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Union
from src.ngrams import NGramModel
from src.preprocessing import tokenize, preprocess_text

@dataclass
class Prediction:
    word: str
    probability: float

    @property
    def confidence_pct(self) -> float:
        return self.probability * 100.0

class BayesianWordPredictor:
    def __init__(
        self, 
        alpha: float = 1.0, 
        higher_order_weight: Optional[float] = None, 
        lower_order_weight: Optional[float] = None
    ):
        if alpha <= 0:
            raise ValueError("Alpha deve ser maior que 0.")

        if higher_order_weight is not None or lower_order_weight is not None:
            w1 = higher_order_weight if higher_order_weight is not None else 0.0
            w2 = lower_order_weight if lower_order_weight is not None else 0.0
            if abs((w1 + w2) - 1.0) > 1e-5:
                raise ValueError("A soma dos pesos deve ser igual a 1.0.")

        self.alpha = alpha
        self.higher_order_weight = higher_order_weight
        self.lower_order_weight = lower_order_weight
        self.model = NGramModel()

    def fit(self, corpus_or_sentences: any) -> BayesianWordPredictor:
        if isinstance(corpus_or_sentences, str):
            doc = preprocess_text(corpus_or_sentences)
            self.model.fit(doc)
        else:
            self.model.fit(corpus_or_sentences)
        return self

    def next_word_distribution(self, context_tokens: List[str]) -> Dict[str, float]:
        vocab = list(self.model.vocab)
        if not vocab:
            return {}

        vocab_size = len(vocab)
        total_unigrams = sum(self.model._unigram_counts.values()) or 1
        
        max_ctx = self.model.order - 1
        ctx = tuple(context_tokens[-max_ctx:]) if context_tokens else ()

        scores: Dict[str, float] = {}
        total_score = 0.0

        for w in vocab:
            count_w = self.model.unigram_count(w)
            p_w = (count_w + self.alpha) / (total_unigrams + self.alpha * vocab_size)

            # Verifica correspondência de n-gram na ordem mais alta
            ngram = ctx + (w,)
            count_ngram = self.model.ngram_count(ngram)
            
            # Se o contexto existir no corpus para a palavra w, aplica o boost da sequência
            if count_ngram > 0:
                p_ctx_given_w = (count_ngram + self.alpha) / (count_w + (self.alpha * vocab_size / 100.0))
            else:
                p_ctx_given_w = (count_ngram + self.alpha) / (count_w + self.alpha * vocab_size)

            score = p_ctx_given_w * p_w
            scores[w] = score
            total_score += score

        if total_score == 0:
            uniform = 1.0 / vocab_size
            return {w: uniform for w in vocab}

        return {w: s / total_score for w, s in scores.items()}

    def predict_word(self, phrase_or_tokens: Union[str, List[str]]) -> Prediction:
        if isinstance(phrase_or_tokens, str):
            tokens = tokenize(phrase_or_tokens)
        else:
            tokens = phrase_or_tokens

        dist = self.next_word_distribution(tokens)
        if not dist:
            return Prediction(word="<desconhecido>", probability=0.0)

        best_word = max(dist, key=dist.get)
        return Prediction(word=best_word, probability=dist[best_word])

    def predict_top_k(self, phrase_or_tokens: Union[str, List[str]], top_k: int = 3) -> List[Prediction]:
        if isinstance(phrase_or_tokens, str):
            tokens = tokenize(phrase_or_tokens)
        else:
            tokens = phrase_or_tokens

        dist = self.next_word_distribution(tokens)
        if not dist:
            return [Prediction(word="<desconhecido>", probability=0.0)]

        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return [Prediction(word=w, probability=p) for w, p in sorted_dist[:top_k]]

BayesianPredictor = BayesianWordPredictor