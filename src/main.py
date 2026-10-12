from __future__ import annotations

import argparse
import os
from typing import Tuple
from src.preprocessing import load_corpus, TextPreprocessor
from src.bayes import BayesianWordPredictor

def build_predictor(corpus_path: str = os.path.join("data", "corpus.txt"), alpha: float = 1.0) -> Tuple[BayesianWordPredictor, TextPreprocessor]:
    predictor = BayesianWordPredictor(alpha=alpha)
    if os.path.exists(corpus_path):
        corpus_text = load_corpus(corpus_path)
        predictor.fit(corpus_text)
    preprocessor = TextPreprocessor()
    return predictor, preprocessor

def main():
    parser = argparse.ArgumentParser(description="Sistema Preditivo de Palavras via Bayes")
    parser.add_argument("frase", type=str, nargs="?", default="O aluno estudou para a prova de", help="Frase de entrada")
    parser.add_argument("-k", "--top", type=int, default=1, help="Quantidade de sugestões no ranking")
    parser.add_argument("-c", "--corpus", type=str, default=os.path.join("data", "corpus.txt"), help="Caminho do corpus")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="Suavização de Laplace")
    
    args = parser.parse_args()

    predictor, _ = build_predictor(corpus_path=args.corpus, alpha=args.alpha)
    predictions = predictor.predict_top_k(args.frase, top_k=args.top)

    for pred in predictions:
        word_formatted = pred.word.capitalize()
        print(f"Sugestão: {word_formatted}, Confiança: {pred.confidence_pct:.2f}%")

if __name__ == "__main__":
    main()