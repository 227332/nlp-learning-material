from typing import List, Any

import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds
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
        term_topic_matrix, sigma_vector, topic_sentence_matrix = self.apply_latent_semantic_analysis(
            preprocessed_sentences,
            sparse_mode=True
        )
        sentence_scores = self._compute_squared_salience_scores(sigma_vector, topic_sentence_matrix)
        # sort sentence indices from higher to lower score and select the top indices
        ranked_sentence_indices = np.argsort(-sentence_scores)
        top_sentence_indices = ranked_sentence_indices[:summary_length]
        # sort the sentences in the order they appear in the original text
        top_sentence_indices.sort()
        return [sentences[index] for index in top_sentence_indices]

    @staticmethod
    def apply_latent_semantic_analysis(sentences, sparse_mode=True):
        """Apply LSA to a list of sentences.
        Support both dense and sparse mode:
        - dense mode uses numpy's SVD implementation
        - sparse mode uses scipy's sparse SVD implementation.
        These 2 implementations return different results, because dense mode applies a full SVD and so computes all
        singular values (i.e. min(A.shape)), while sparse mode only supports partial SVD with number of singular values
        to be at most equal to min(A.shape)-1.

        Since singular values represent the hidden topics of the sentences in the context of text summarization,
        using a different number of hidden topics affects the final importance ranking of the sentences.
        None of this approach is optimal, a better approach would be to let the user specify the number of topics k as
        input. In this way:
        - the user can run LSASummarizer with different values of k and apply some coherence metric or qualitative
        analysis on the results to select the best k, or
        - the user can use his/her prior knowledge of the text to provide a good value of k.
        This recommended approach is not supported in this code because letting the user specify the number of topics
        goes beyond the paper used as reference for this implementation.
        """
        term_sentence_matrix = LSASummarizer._compute_term_sentence_matrix(sentences)
        if sparse_mode is True:
            # scipy's svds(A,k) requires 1 <= k < min(A.shape)
            num_topics = min(term_sentence_matrix.shape) - 1
            term_topic_matrix, sigma_vector, topic_sentence_matrix = svds(
                term_sentence_matrix,
                k=num_topics,
                which='LM'
            )
        else:
            # convert sparse matrix to a dense matrix (2D array)
            term_sentence_matrix_dense = term_sentence_matrix.toarray()
            term_topic_matrix, sigma_vector, topic_sentence_matrix = svd(
                term_sentence_matrix_dense,
                full_matrices=False
            )
        return term_topic_matrix, sigma_vector, topic_sentence_matrix

    @staticmethod
    def _compute_term_sentence_matrix(sentences: List[str]) -> Any:
        """Compute the weighted term-sentence matrix.
        The matrix is returned in its CSR sparse representation
        """
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=None)
        sentence_term_matrix = vectorizer.fit_transform(sentences)
        return sentence_term_matrix.transpose()

    @staticmethod
    def _compute_squared_salience_scores(sigma_vector: Any, topic_sentence_matrix: Any) -> Any:
        """Compute the squared value of the salience score of each sentence.
        The salience score is also called sentence length in the literature
        """
        squared_scores = np.dot(np.square(sigma_vector), np.square(topic_sentence_matrix))
        return squared_scores
