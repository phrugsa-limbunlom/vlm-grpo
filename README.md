# VLM-GRPO: Vision-Language Model Post-Training with Group Relative Policy Optimization (GRPO)

A PyTorch implementation for training Vision-Language Models (VLMs) using GRPO (Group Relative Policy Optimization) with LoRA fine-tuning for efficient training.

## Features

- **GRPO Training**: Implements Group Relative Policy Optimization for training VLMs
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using Low-Rank Adaptation
- **Multimodal Support**: Handles both text and image inputs for vision-language tasks
- **Reward Functions**: Multiple reward mechanisms including format compliance and mathematical accuracy
- **Flexible Evaluation**: Support for various evaluation datasets and metrics
- **Hugging Face Integration**: Seamless integration with Hugging Face models and datasets

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/vlm-grpo.git
cd vlm-grpo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt

# Install PyTorch matching your CUDA/CPU setup (example for CUDA 12.6):
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

4. Login to your Huggingface account
```
huggingface-cli login
```
## Dependencies

The project uses the following key dependencies:

- **Core ML Libraries**: `torch`, `transformers`, `datasets`
- **PEFT**: `peft` for LoRA fine-tuning
- **TRL**: `trl` for GRPO training
- **Math Verification**: `math-verify`, `latex2sympy2-extended`
- **Qwen VL Utils**: `qwen_vl_utils` for vision-language processing

## Project Structure

```
vlm-grpo/
├── src/
│   ├── main.py              # Main training and evaluation script
│   ├── vlm_grpo.py          # VLM-GRPO trainer implementation
│   ├── custom_datasets.py   # Dataset loading and transformation utilities
│   ├── reward.py            # Reward function implementations
│   └── constants.py         # Configuration constants
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Quick Start

### Training a Model

```bash
python src/main.py --mode train
```

### Testing a Trained Model

```bash
python src/main.py --mode test
```

### Evaluating on a Dataset

```bash
python src/main.py --mode eval
```

## Configuration

The project uses configuration classes defined in `src/constants.py`:

### Model Configuration
- **Base Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Dataset**: `lmms-lab/multimodal-open-r1-8k-verified`
- **Output Directory**: `Qwen2.5-VL-3B-Instruct-Thinking`

### Training Parameters
- **Learning Rate**: 1e-5
- **Epochs**: 1
- **Batch Size**: 2
- **Max Length**: 1024
- **LoRA Parameters**: r=8, alpha=32, dropout=0.1

## Key Components

### VLMGRPO Class
The main trainer class that handles:
- Model loading with LoRA configuration
- GRPO training setup
- Training execution with reward functions

### Reward Functions
Two primary reward mechanisms:
1. **Format Reward**: Ensures outputs follow `<think>...</think><answer>...</answer>` format
2. **Accuracy Reward**: Mathematical verification using LaTeX parsing and verification

### Dataset Handling
- Automatic dataset loading from Hugging Face Hub
- Data transformation for multimodal inputs
- Train/test splitting with reproducible seeds

## Evaluation

The project supports evaluation on various datasets:

- **MathVista**: Mathematical reasoning with visual inputs
- **Custom Datasets**: Any Hugging Face dataset with image and text fields

### Evaluation Metrics
- **Exact Match Accuracy**: String comparison for exact answers
- **Mathematical Accuracy**: LaTeX-based mathematical verification
- **Format Compliance**: Adherence to expected output format

## Advanced Usage

### Custom Reward Functions
You can implement custom reward functions by extending the `Reward` class:

```python
class CustomReward:
    @staticmethod
    def custom_reward(completions, **kwargs):
        # Your custom reward logic
        return rewards
```

### Custom Datasets
To use your own dataset, modify the `ModelConfig.DATASET_ID` in `constants.py`:

```python
class ModelConfig:
    DATASET_ID = "your-username/your-dataset"
```

## Training Process

1. **Data Loading**: Loads and splits the dataset (80% train, 20% test)
2. **Model Preparation**: Loads base model and applies LoRA configuration
3. **GRPO Setup**: Configures training parameters and reward functions
4. **Training**: Executes GRPO training with logging and checkpointing
5. **Model Saving**: Saves final model and pushes to Hugging Face Hub

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `TrainingConfig.BATCH_SIZE`
2. **Model Loading Errors**: Ensure you have sufficient disk space and internet connection
3. **Dataset Loading Issues**: Check dataset ID and availability on Hugging Face Hub

### Logging

Training logs are saved to `./output/` directory with timestamps for easy tracking.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base VLM model
- [Hugging Face](https://huggingface.co/) for the transformers library
- [TRL](https://github.com/huggingface/trl) for GRPO implementation
- [PEFT](https://github.com/huggingface/peft) for LoRA fine-tuning

---

**Note**: This project is designed for research purposes. Please ensure you have appropriate computational resources and follow responsible AI practices when training and deploying models.