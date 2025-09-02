from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from typing import Optional, Dict, Any, List, Union


class Dataset:
    """
    A class for loading and splitting datasets for training and testing.
    
    This class provides functionality to load datasets from Hugging Face Hub
    and split them into training and testing subsets.
    """

    def __init__(self, dataset_id: str) -> None:
        """
        Initialize the dataset loader with a specific dataset ID.
        
        Args:
            dataset_id (str): The identifier of the dataset on Hugging Face Hub
        """
        self.dataset_id: str = dataset_id
        self.split_dataset: Optional[DatasetDict] = None

    def load_data(self) -> None:
        """
        Load the dataset and split it into training and testing sets.
        
        Loads the first 5% of the dataset and splits it with 80% training
        and 20% testing using a fixed random seed for reproducibility.
        """
        dataset: HFDataset = load_dataset(self.dataset_id, split='train[:5%]')

        self.split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    def get_train_dataset(self) -> HFDataset:
        """
        Get the training dataset split.
        
        Returns:
            The training portion of the dataset
        """
        if self.split_dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        return self.split_dataset['train']

    def get_test_dataset(self) -> HFDataset:
        """
        Get the testing dataset split.
        
        Returns:
            The testing portion of the dataset
        """
        if self.split_dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        return self.split_dataset['test']


class TransformData:
    """
    A class for transforming raw dataset examples into the format required by the VLM model.
    
    This class handles the conversion of multimodal inputs (images + text) into
    conversation format with proper system prompts and chat templates.
    """

    def __init__(self, model_id: str, processor: Any) -> None:
        """
        Initialize the data transformer with model and processor.
        
        Args:
            model_id (str): The identifier of the model being used
            processor: The processor for handling multimodal inputs
        """
        self.model_id: str = model_id
        self.processor: Any = processor

    def _make_conversation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single example into conversation format.
        
        Creates a conversation structure with system prompt, user input (image + problem),
        and applies the chat template for the specific model.
        
        Args:
            example: A single dataset example containing 'problem' and 'image' fields
            
        Returns:
            dict: Transformed example with 'prompt' and 'image' keys
        """
        SYSTEM_PROMPT: str = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )

        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["problem"]},
                ],
            },
        ]

        prompt: str = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        return {
            "prompt": prompt,
            "image": example["image"]
        }

    def transform(self, dataset: HFDataset) -> HFDataset:
        """
        Transform an entire dataset using the conversation format.
        
        Applies the conversation transformation to all examples in the dataset.
        
        Args:
            dataset: The dataset to transform
            
        Returns:
            The transformed dataset with conversation format applied
        """
        return dataset.map(self._make_conversation)