import torch
from transformers import Qwen2_5_VLForConditionalGeneration, PreTrainedProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset

from reward import Reward


class LoRA:
    """
    A class for configuring and applying LoRA (Low-Rank Adaptation) to models.
    
    LoRA is a technique that reduces the number of trainable parameters by adding
    low-rank matrices to existing layers, making fine-tuning more efficient.
    """

    def __init__(self, model: Qwen2_5_VLForConditionalGeneration) -> None:
        """
        Initialize the LoRA configuration with a base model.
        
        Args:
            model: The base model to apply LoRA to
        """
        self.model: Qwen2_5_VLForConditionalGeneration = model

    def config_lora(self, task_type: str, r: int, alpha: int, dropout: float) -> PeftModel:
        """
        Configure and apply LoRA to the model.
        
        Args:
            task_type (str): The type of task (e.g., "CAUSAL_LM" for causal language modeling)
            r (int): The rank of the low-rank matrices
            alpha (int): The scaling factor for LoRA weights
            dropout (float): Dropout probability for LoRA layers
            
        Returns:
            The model with LoRA applied
        """
        lora_config: LoraConfig = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
        )

        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()

        return self.model


class VLMGRPO:
    """
    Vision-Language Model (VLM) training using GRPO (Generative Reward-Powered Optimization).
    
    This class provides functionality for loading, configuring, and training VLM models
    using the GRPO algorithm with LoRA fine-tuning for efficiency.
    """

    def __init__(self, model_id: str, processor: PreTrainedProcessor) -> None:
        """
        Initialize the VLM-GRPO trainer.
        
        Args:
            model_id (str): The identifier of the pre-trained model to load
            processor: The processor for handling multimodal inputs (text + images)
        """
        self.model_id: str = model_id
        self.processor: PreTrainedProcessor = processor
        self.model: Optional[PeftModel] = None
       
    def load_model(self) -> None:
        """
        Load the pre-trained VLM model and apply LoRA configuration.
        
        Loads a Qwen2.5-VL model with bfloat16 precision and applies LoRA
        with default hyperparameters (r=8, alpha=32, dropout=0.1).
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.model = LoRA(self.model).config_lora(task_type="CAUSAL_LM", r=8, alpha=32, dropout=0.1)

    def _config_grpo(self,
                     output_dir: str,
                     lr: float,
                     epochs: int,
                     batch_size: int,
                     max_length: int,
                     num_generations: int,
                     max_prompt_length: int,
                     log_steps: int,
                     save_steps: int) -> GRPOConfig:
        """
        Configure GRPO training parameters.
        
        Args:
            output_dir (str): Directory to save model checkpoints and logs
            lr (float): Learning rate for training
            epochs (int): Number of training epochs
            batch_size (int): Batch size per device
            max_length (int): Maximum length of generated completions
            num_generations (int): Number of generations per prompt for reward calculation
            max_prompt_length (int): Maximum length of input prompts
            log_steps (int): Frequency of logging training metrics
            save_steps (int): Frequency of saving model checkpoints
            
        Returns:
            GRPOConfig: Configured training arguments
        """
        # Configure training arguments using GRPOConfig
        training_args: GRPOConfig = GRPOConfig(
            output_dir=output_dir,
            learning_rate=lr,
            remove_unused_columns=False,  # to access the solution column in accuracy_reward
            num_train_epochs=epochs,
            bf16=True,

            # Parameters that control the data preprocessing
            per_device_train_batch_size=batch_size,
            max_completion_length=max_length,  # default: 256
            num_generations=num_generations,  # default: 8
            max_prompt_length=max_prompt_length,

            # Parameters related to reporting and saving
            report_to=["tensorboard"],
            logging_steps=log_steps,
            push_to_hub=True,
            save_strategy="steps",
            save_steps=save_steps,
        )
        
        return training_args

    def train(self, 
                train_dataset: HFDataset, 
                output_dir: str, 
                learning_rate: float, 
                epochs: int, 
                batch_size: int, 
                max_length: int, 
                num_generations: int, 
                max_prompt_length: int, 
                log_steps: int, 
                save_steps: int) -> None:
        """
        Train the VLM model using GRPO algorithm.
        
        Args:
            train_dataset (HFDataset): The training dataset to use for training
            output_dir (str): Directory path where training outputs will be saved
            learning_rate (float): Learning rate for the optimizer during training
            epochs (int): Number of complete training epochs to run
            batch_size (int): Number of samples per training batch per device
            max_length (int): Maximum length of generated completions during training
            num_generations (int): Number of generations per prompt for reward calculation
            max_prompt_length (int): Maximum length of input prompts during training
            log_steps (int): Frequency of logging training metrics (every N steps)
            save_steps (int): Frequency of saving model checkpoints (every N steps)
            
        Note:
            This method configures and executes GRPO training with the specified hyperparameters.
            The training process will save checkpoints and logs to the specified output directory.
        """
        train_args: GRPOConfig = self._config_grpo(output_dir=output_dir,
                                       lr=learning_rate,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       max_length=max_length,
                                       num_generations=num_generations,
                                       max_prompt_length=max_prompt_length,
                                       log_steps=log_steps,
                                       save_steps=save_steps)

        trainer: GRPOTrainer = GRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            reward_funcs=[Reward.format_reward, Reward.accuracy_reward],
            args=train_args,
            train_dataset=train_dataset,
        )

        trainer.train()

        trainer.save_model(train_args.output_dir)
        trainer.push_to_hub(dataset_name=output_dir)
