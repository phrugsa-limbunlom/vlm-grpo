from custom_datasets import Dataset, TransformData
from vlm_grpo import VLMGRPO
from transformers import AutoProcessor


if __name__ == "__main__":

    dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir = "Qwen2.5-VL-3B-Instruct-Thinking"

    learning_rate=1e-5
    epochs=1
    batch_size=2
    max_length=1024
    num_generations=2,
    max_prompt_length=2048,
    log_steps=10,
    save_steps=10

    train_dataset = Dataset(dataset_id=dataset_id).get_train_dataset()

    train_dataset = TransformData().transform(train_dataset)


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