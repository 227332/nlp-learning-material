from abc import ABC, abstractmethod


class BaseSummarizer(ABC):

    def __init__(self, language: str):
        self.language = language

    @abstractmethod
    def summarize(self, text: str, summary_length: int):
        pass
