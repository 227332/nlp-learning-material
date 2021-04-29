import re
import string
from typing import List, Any

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


class NLPHelper:
    """Helper class for typical NLP tasks"""

    def __init__(self, language: str = "english", stemming: bool = True) -> None:
        self._language = language
        self._stopwords = set(stopwords.words(language))
        if stemming:
            self._stemmer = SnowballStemmer(language)
        else: 
            self._stemmer = None

    @property
    def language(self) -> str:
        return self._language

    @property
    def stopwords(self) -> List[str]:
        return self._stopwords

    @property
    def stemmer(self) -> Any:
        return self._stemmer    

    def preprocess_text(self, text: str) -> str: 
        """Pre-process input text with conversion to lowercase, punctuation removal,
        stopwords removal and stemming
        """
        text_without_punctuation = self.remove_punctuation(text.lower())
        tokens = word_tokenize(text_without_punctuation)
        tokens_without_stopwords = self.remove_stopwords(tokens)
        preprocessed_tokens = [self.stem_token(token) for token in tokens_without_stopwords]
        preprocessed_text = ' '.join(preprocessed_tokens)
        return preprocessed_text

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from a list of tokens"""
        return [token for token in tokens if token not in self._stopwords]

    def stem_token(self, token: str) -> str:
        """Perform stemming on a token"""
        return self._stemmer.stem(token)

    def get_sentences(self, text: str) -> List[str]:
        """Perform sentence tokenization of input text"""
        return sent_tokenize(text, language=self._language)

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from input text"""
        text_with_whitespace_replaced_by_space = re.sub("\s+", " ", text).strip()
        punctuation_characters = string.punctuation + '’'  # ’ not ascii character
        regex_pattern = "[{}]+".format(re.escape(punctuation_characters))
        text_without_punctuation = re.sub(regex_pattern, '', text_with_whitespace_replaced_by_space)
        return text_without_punctuation
