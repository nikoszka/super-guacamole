from abc import ABC, abstractmethod
from typing import List, Text


STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', 'Question:', 'Context:', 'Answer:']
# '\nAnswer ' catches prompt leakage where the model re-emits the BRIEF instruction
STOP_SEQUENCES_DETAILED = ['\n\n\n\n', 'Question:', 'Context:', 'Answer:', '\nAnswer ']


class BaseModel(ABC):

    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        pass
