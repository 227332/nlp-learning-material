from typing import List, Any

import numpy as np
from numpy.linalg import svd
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers.NLPHelper import NLPHelper
from summarizers.BaseSummarizer import BaseSummarizer


class LSASummarizer(BaseSummarizer):
    """Apply automatic text summarization using Latent Semantic Analysis (LSA),
    as described in the paper 'Using Latent Semantic Analysis in Text Summarization
    and Summary Evaluation' (Steinberger et al., 2004):
    http://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf
    """

    def __init__(self, language: str) -> None:
        super().__init__(language)
        self._nlp_helper = NLPHelper(language)

    def summarize(self, text: str, summary_length: int = 3) -> List[str]:
        """Summarize input text with a desired number of sentences"""
        sentences = self._nlp_helper.get_sentences(text)
        text_length = len(sentences)
        if summary_length >= text_length or text_length == 0:
            return sentences
        if summary_length <= 0:
            return []
        preprocessed_sentences = [self._nlp_helper.preprocess_text(sentence) for sentence in sentences]
        term_sentence_matrix = self._compute_term_sentence_matrix(preprocessed_sentences)
        term_topic_matrix, sigma_vector, topic_sentence_matrix = svd(term_sentence_matrix, full_matrices=False)
        sentence_scores = self._compute_squared_salience_scores(sigma_vector, topic_sentence_matrix)
        # sort sentence indices from higher to lower score and select the top indices
        ranked_sentence_indices = np.argsort(-sentence_scores)
        top_sentence_indices = ranked_sentence_indices[:summary_length]
        # sort the sentences in the order they appear in the original text
        top_sentence_indices.sort()
        return [sentences[index] for index in top_sentence_indices]

    @staticmethod
    def _compute_term_sentence_matrix(sentences: List[str]) -> Any:
        """Compute the weighted term-sentence matrix"""
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=None)
        matrix = vectorizer.fit_transform(sentences)
        return matrix

    @staticmethod
    def _compute_squared_salience_scores(sigma_vector: Any, topic_sentence_matrix: Any) -> Any:
        """Compute the squared value of the salience score of each sentence.
        The salience score is also called sentence length in the literature
        """
        squared_scores = np.dot(np.square(sigma_vector), np.square(topic_sentence_matrix))
        return squared_scores
