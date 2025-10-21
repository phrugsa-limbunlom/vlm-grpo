from datetime import time
from datasets import load_dataset
from custom_datasets import TransformData
from vlm_grpo import VLMGRPO, logger
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import time
import torch
from qwen_vl_utils import process_vision_info

from custom_datasets import Dataset
from constants import SystemPrompts, TrainingConfig, ModelConfig
from typing import Optional
import argparse

from PIL import Image

def train():

    output_dir = ModelConfig.OUTPUT_DIR

    learning_rate = TrainingConfig.LEARNING_RATE
    epochs = TrainingConfig.EPOCHS
    batch_size = TrainingConfig.BATCH_SIZE
    max_length = TrainingConfig.MAX_LENGTH
    num_generations = TrainingConfig.NUM_GENERATIONS
    max_prompt_length = TrainingConfig.MAX_PROMPT_LENGTH
    log_steps = TrainingConfig.LOG_STEPS
    save_steps = TrainingConfig.SAVE_STEPS

    train_dataset = dataset.get_train_dataset()

    train_dataset = TransformData(model_id=model_id, processor=AutoProcessor.from_pretrained(model_id)).transform(
        train_dataset)

    vlm_grpo = VLMGRPO(model_id=model_id,
                       processor=AutoProcessor.from_pretrained(model_id))

    vlm_grpo.load_model()

    vlm_grpo.train(train_dataset=train_dataset,
                   output_dir=output_dir,
                   learning_rate=learning_rate,
                   epochs=epochs,
                   batch_size=batch_size,
                   max_length=max_length,
                   num_generations=num_generations,
                   max_prompt_length=max_prompt_length,
                   log_steps=log_steps,
                   save_steps=save_steps)

def test():
    trained_model_id = ModelConfig.TRAINED_MODEL_ID
    trained_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        trained_model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    trained_processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

    test_dataset = dataset.get_test_dataset()

    def generate_with_reasoning(problem, image):
        SYSTEM_PROMPT: str = SystemPrompts.SYSTEM_PROMPT

        # Conversation setting for sending to the model
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": problem},
                ],
            },
        ]
        prompt = trained_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # Process images using the process_vision_info from qwen_vl_utils
        image_inputs, video_inputs = process_vision_info(conversation)

        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(trained_model.device)

        # Generate text without gradients
        start_time = time.time()
        with torch.no_grad():
            output_ids = trained_model.generate(**inputs, max_new_tokens=500)
        end_time = time.time()

        # Decode and extract model response
        generated_text = trained_processor.decode(output_ids[0], skip_special_tokens=True)

        # Get inference time
        inference_duration = end_time - start_time

        # Get number of generated tokens
        num_input_tokens = inputs["input_ids"].shape[1]
        num_generated_tokens = output_ids.shape[1] - num_input_tokens

        return generated_text, inference_duration, num_generated_tokens

    generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(test_dataset[0]['problem'],
                                                                                       test_dataset[0]['image'])

    logger.info(generated_text)


def evaluate(data_id : str, split: str, limit: Optional[int] = None):
    """
    Evaluate the trained VLM on a Hugging Face dataset split and report simple exact-match accuracy.

    Args:
        data_id (str): Hugging Face dataset ID to evaluate on (e.g., "AI4Math/MathVista").
        split (str): Dataset split to use (e.g., "testmini" or "test" for MathVista).
        limit (Optional[int]): If provided, evaluate only the first N examples.

    Behavior:
        - Loads the trained model from `ModelConfig.TRAINED_MODEL_ID`.
        - Loads the dataset split via `datasets.load_dataset(data_id)[split]`.
        - Builds a multimodal conversation (image + question/query), applies the chat template,
          and runs generation.
        - Normalizes predictions and gold answers by `answer_type` and `precision` when present
          (e.g., integer rounding, fixed decimals for floats).
        - Logs running accuracy every 50 examples and final accuracy at the end.

    """
    trained_model_id = ModelConfig.TRAINED_MODEL_ID
    trained_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        trained_model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    trained_processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")
    base_processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

    dataset = load_dataset(data_id)[split]

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    def normalize(pred: str, ex: dict) -> str:
        atype = ex.get("answer_type", "text")

        if atype == "integer":
            try:
                return str(int(round(float(pred.strip()))))
            except:
                return pred.strip().lower()
        if atype == "float":
            precision = ex.get("precision", 1)

            try:
                # Extract numeric value from the prediction, handling various formats
                pred_clean = pred.strip()
                # Try to find the last number in the string
                import re
                numbers = re.findall(r'-?\d+\.?\d*', pred_clean)
                if numbers:
                    val = float(numbers[-1].replace(",",""))
                    fmt = f"{{:.{int(precision)}f}}" # format
                    return fmt.format(val)
                else:
                    return pred_clean.lower()
            except:
                return pred.strip().lower()
        return pred.strip().lower()

    def normalize_gold(ans: str, ex: dict) -> str:
        atype = ex.get("answer_type", "text")
        if atype == "integer":
            try:
                return str(int(ans.strip))
            except:
                return ans.strip().lower()
        if atype == "float":
            precision = ex.get("precision",1)
            try:
                val = float(ans.strip())
                fmt = f"{{:.{int(precision)}f}}"
                return fmt.format(val)
            except:
                return ans.strip().lower()
        return ans.strip().lower()

    
    correct = 0
    total = 0

    for example in dataset:
        question = example.get("query") or example.get("question") or ""
        pil_image = example.get("decoded_image")
        if pil_image is None:
            img_path = example.get("image")
            pil_image = Image.open(img_path).convert("RGB")

        conversation = [
            {"role": "system", "content": SystemPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = trained_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = base_processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(trained_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = trained_model.generate(**inputs, max_new_tokens=256)
        generated_text = trained_processor.decode(output_ids[0], skip_special_tokens=True)

        pred_norm = normalize(generated_text, example)
        gold_norm = normalize_gold(example.get("answer",""), example)

        correct += int(pred_norm == gold_norm)
        total += 1

        if total % 50 == 0:
            logger.info(f"Evaluated {total} examples, running accuracy: {correct/total:.4f}")

    acc = correct / max(total, 1)
    logger.info(f"{data_id} ({split}) accuracy: {acc:.4f} ({correct}/{total})")
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VLM GRPO Training and Testing")
    parser.add_argument("--mode", choices=["train", "test", "eval"], default="train", 
                       help="Mode to run: train, test, or evaluate")
    
    args = parser.parse_args()

    dataset_id = ModelConfig.DATASET_ID
    model_id = ModelConfig.MODEL_ID

    dataset = Dataset(dataset_id=dataset_id)

    dataset.load_data()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "eval":
        dataset = ModelConfig.EVAL_DATASET_ID
        split = ModelConfig.EVAL_DATASET_SPLIT
        evaluate(dataset, split, 100)