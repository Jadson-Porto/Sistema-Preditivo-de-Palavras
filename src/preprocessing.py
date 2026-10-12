from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List

@dataclass
class Document:
    sentences: List[List[str]] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return sum(len(s) for s in self.sentences)

    def __iter__(self):
        return iter(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

class TextPreprocessor:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        # Trata hífens transformando em espaço (ex: n-grams -> n grams)
        text = text.replace("-", " ")
        # Mantém pontuação de fim de frase para separação (.!?)
        text = re.sub(r'[^a-zA-Záàâãéèêíïóôõöúçñ\s\.\!\?]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        cleaned = re.sub(r'[\.\!\?]', '', cleaned)
        return cleaned.split() if cleaned else []

    def process_document(self, text: str) -> Document:
        cleaned = self.clean_text(text)
        raw_sentences = re.split(r'[\.\!\?]+', cleaned)
        sentences = []
        for s in raw_sentences:
            tokens = s.strip().split()
            if tokens:
                sentences.append(tokens)
        return Document(sentences=sentences)

def load_corpus(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def preprocess_text(text: str) -> Document:
    processor = TextPreprocessor()
    return processor.process_document(text)

def tokenize(text: str) -> List[str]:
    processor = TextPreprocessor()
    return processor.tokenize(text)