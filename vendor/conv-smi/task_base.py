from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json

class Task(ABC):
    """Base class for all tasks (layered_vqa, etc.)"""

    def __init__(self, version: str):
        self.version: str = version
        self.task_name: str = self.get_task_name()
        self.answer_description: str = self.get_answer_description()

    @abstractmethod
    def get_task_name(self) -> str:
        """Return the name of the task (e.g., 'layered_vqa')"""

    @abstractmethod
    def get_dataset_file(self) -> str:
        """Return the name of the file containing the samples for the task"""
        pass

    @abstractmethod
    def get_samples(self, filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a list of samples for the task"""
        pass

    def get_sample(self, sample_id: str) -> Dict[str, Any]:
        samples = self.get_samples()
        id2samples = {sample["task_id"]: sample for sample in samples}
        if sample_id not in id2samples:
            raise ValueError(f"Sample ID {sample_id} not found")
        return id2samples[sample_id]

    @abstractmethod
    def get_answer_description(self) -> str:
        """Return description of what constitutes a valid answer"""
        pass

    @abstractmethod
    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate the system prompt for the given sample"""
        pass

    @abstractmethod
    def evaluate_conversation(self, trace: List[Dict[str, Any]], sample: Dict[str, Any]) -> Dict[str, Any]:
        """Grade a finished conversation, once, from its whole trace.

        Returns a dict carrying "score" and/or "is_correct"; the simulators
        assert on that. Grading is conversation-level rather than turn-level,
        so there is no answer-extraction step in front of it."""
        pass

    @abstractmethod
    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate the populated prompt for fully-specified attempts"""
        pass

    @abstractmethod
    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate the populated prompt for concatenated experiment"""
        pass

    def save_samples(self, samples: List[Dict[str, Any]]):
        dataset_fn = self.get_dataset_file()
        with open(dataset_fn, "w") as f:
            json.dump(samples, f, indent=4)


    @abstractmethod
    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Given a sample in the dataset file, return a dictionary with all the information from the original sample; helpful for displaying the sample in the annotation UI"""
        pass
