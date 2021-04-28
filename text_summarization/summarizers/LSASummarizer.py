import re
import string
from typing import List, Any

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from numpy.linalg import svd
from sklearn.feature_extraction.text import TfidfVectorizer

from summarizers.BaseSummarizer import BaseSummarizer


class LSASummarizer(BaseSummarizer):
    """Apply automatic text summarization using Latent Semantic Analysis (LSA),
    as described in the paper 'Using Latent Semantic Analysis in Text Summarization 
    and Summary Evaluation' (Steinberger et al., 2004): 
    http://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf
    """

    def __init__(self, language: str) -> None:
        super().__init__(language)
        self._stemmer = SnowballStemmer(language)
        self._stopwords = set(stopwords.words(language))

    def summarize(self, text: str, summary_length: int = 3) -> List[str]:
        """Summarize input text with a desired number of sentences"""
        sentences = sent_tokenize(text, language=self.language)
        text_length = len(sentences)
        if summary_length >= text_length or text_length==0:
            return sentences
        if summary_length <= 0:
            return []
        preprocessed_sentences = [self._preprocess_text(sentence) for sentence in sentences]
        term_sentence_matrix = self._compute_term_sentence_matrix(preprocessed_sentences)
        term_topic_matrix, sigma_vector, topic_sentence_matrix = singular_value_decomposition(term_sentence_matrix, full_matrices=False)
        sentence_scores = self._compute_squared_salience_scores(sigma_vector, topic_sentence_matrix)
        # sort sentence indices from higher to lower score and select the top indices
        ranked_sentence_indices = np.argsort(-sentence_scores)
        top_sentence_indices = ranked_sentence_indices[: summary_length]
        # sort the sentences in the order they appear in the original text
        top_sentence_indices.sort()
        return [sentences[index] for index in top_sentence_indices]

    def _preprocess_text(self, text: str) -> str: 
        """Pre-process input text with conversion to lowercase, punctuation removal,
        stopwords removal and stemming
        """
        text_without_punctuation = self._remove_punctuation(text.lower())
        tokens = word_tokenize(text_without_punctuation)
        tokens_without_stopwords = self._remove_stopwords(tokens)
        preprocessed_tokens = [self._stem_token(token) for token in tokens_without_stopwords]
        preprocessed_text = ' '.join(preprocessed_tokens)
        return preprocessed_text

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from input text"""
        text_with_whitespace_replaced_by_space = re.sub("\s+", " ", text).strip()
        punctuation_characters = string.punctuation + '’'  # ’ not ascii character
        regex_pattern = "[{}]+".format(re.escape(punctuation_characters))
        text_without_punctuation = re.sub(regex_pattern, '', text_with_whitespace_replaced_by_space)
        return text_without_punctuation

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from a list of tokens"""
        return [token for token in tokens if token not in self._stopwords]

    def _stem_token(self, token: str) -> str:
        """Perform stemming on a token"""
        return self._stemmer.stem(token)

    def _compute_term_sentence_matrix(self, sentences: List[str]) -> Any:
        """Compute the weighted term-sentence matrix"""
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=None)
        matrix = vectorizer.fit_transform(sentences)
        return matrix

    def _compute_squared_salience_scores(self, sigma_vector: Any, topic_sentence_matrix: Any) -> Any:
        """Compute the squared value of the salience score of each sentence.
        The salience score is also called sentence length in the literature
        """
        squared_scores = np.dot(np.square(sigma_vector), np.square(topic_sentence_matrix))
        return squared_scores