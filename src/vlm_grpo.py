import torch
import logging
import os
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset
from typing import Any

from reward import Reward


def setup_logger(output_dir: str = "./output") -> logging.Logger:
    """Minimal logger setup with file and console output."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"vlm_grpo_{output_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger

logger = setup_logger(output_dir="./output")

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
        logger.info("LoRA configuration initialized")

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
        logger.info(f"Configuring LoRA with r={r}, alpha={alpha}, dropout={dropout}")
        logger.info("Target modules: ['q_proj', 'v_proj']")
        
        lora_config: LoraConfig = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
        )

        logger.info("Applying LoRA configuration to model...")
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info("LoRA configuration applied successfully")
        logger.info("Model trainable parameters:")
        self.model.print_trainable_parameters()

        return self.model


class VLMGRPO:
    """
    Vision-Language Model (VLM) training using GRPO (Generative Reward-Powered Optimization).
    
    This class provides functionality for loading, configuring, and training VLM models
    using the GRPO algorithm with LoRA fine-tuning for efficiency.
    """

    def __init__(self, model_id: str, processor: Any, output_dir: str = "./output") -> None:
        """
        Initialize the VLM-GRPO trainer.
        
        Args:
            model_id (str): The identifier of the pre-trained model to load
            processor: The processor for handling multimodal inputs (text + images)
            output_dir (str): Directory for saving outputs and logs
        """
        self.model_id: str = model_id
        self.processor: Any = processor
        self.model: Optional[PeftModel] = None
        self.output_dir: str = output_dir

        logger.info("VLM GRPO trainer initialized")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Output directory: {output_dir}")
       
    def load_model(self) -> None:
        """
        Load the pre-trained VLM model and apply LoRA configuration.
        
        Loads a Qwen2.5-VL model with bfloat16 precision and applies LoRA
        with default hyperparameters (r=8, alpha=32, dropout=0.1).
        """
        logger.info(f"Loading pre-trained model: {self.model_id}")
        logger.info("Using bfloat16 precision and auto device mapping")
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.info("Base model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            raise

        logger.info("Applying LoRA configuration...")
        
        self.model = LoRA(self.model).config_lora(
            task_type="CAUSAL_LM", r=8, alpha=32, dropout=0.1
        )
        logger.info("Model loading and LoRA configuration completed")

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
        logger = logging.getLogger("vlm_grpo")
        logger.info("Configuring GRPO training parameters:")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Num generations: {num_generations}")
        logger.info(f"  Max prompt length: {max_prompt_length}")
        logger.info(f"  Log steps: {log_steps}")
        logger.info(f"  Save steps: {save_steps}")
        
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
        
        logger.info("GRPO configuration completed successfully")
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
        logger = logging.getLogger("vlm_grpo")
        logger.info("=" * 60)
        logger.info("STARTING VLM GRPO TRAINING")
        logger.info("=" * 60)
        
        # Log dataset information
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Dataset features: {train_dataset.features}")
        
        # Configure training
        train_args: GRPOConfig = self._config_grpo(
            output_dir=output_dir,
            lr=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_length,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            log_steps=log_steps,
            save_steps=save_steps
        )

        logger.info("Initializing GRPO trainer...")
        try:
            trainer: GRPOTrainer = GRPOTrainer(
                model=self.model,
                processing_class=self.processor,
                reward_funcs=[Reward.format_reward, Reward.accuracy_reward],
                args=train_args,
                train_dataset=train_dataset,
            )
            logger.info("GRPO trainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GRPO trainer: {str(e)}")
            raise

        logger.info("Starting training...")
        logger.info(f"Training will run for {epochs} epochs")
        logger.info(f"Model checkpoints will be saved every {save_steps} steps")
        logger.info(f"Training metrics will be logged every {log_steps} steps")
        
        try:
            trainer.train()
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

        # Save final model
        logger.info(f"Saving final model to {train_args.output_dir}")
        try:
            trainer.save_model(train_args.output_dir)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

        # Push to hub
        logger.info(f"Pushing model to hub with name: {output_dir}")
        try:
            trainer.push_to_hub(dataset_name=output_dir)
            logger.info("Model pushed to hub successfully")
            
        except Exception as e:
            logger.error(f"Failed to push model to hub: {str(e)}")
            raise

        logger.info("VLM GRPO TRAINING COMPLETED SUCCESSFULLY")
