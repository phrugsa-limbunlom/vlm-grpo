class SystemPrompts:
    """System prompts for different VLM model configurations."""
    
    SYSTEM_PROMPT = ( 
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )


class TrainingConfig:
    """Default training configuration parameters."""
    
    LEARNING_RATE = 1e-5
    EPOCHS = 1
    BATCH_SIZE = 2
    MAX_LENGTH = 1024
    NUM_GENERATIONS = 2
    MAX_PROMPT_LENGTH = 2048
    LOG_STEPS = 10
    SAVE_STEPS = 10


class ModelConfig:
    """Model and dataset configuration constants."""
    
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    DATASET_ID = 'lmms-lab/multimodal-open-r1-8k-verified'
    OUTPUT_DIR = "Qwen2.5-VL-3B-Instruct-Thinking"
    USER_NAME = "PhrugsaL"
    TRAINED_MODEL_ID = USER_NAME + "/" + OUTPUT_DIR

    EVAL_DATASET_ID = "AI4Math/MathVista"
    EVAL_DATASET_SPLIT = "testmini"