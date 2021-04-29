from abc import ABC, abstractmethod


class BaseSummarizer(ABC):

    def __init__(self, language: str) -> None:
        self._language = language

    @property
    def language(self) -> str:
        return self._language

    @abstractmethod
    def summarize(self, text: str, summary_length: int) -> List[str]:
        pass
