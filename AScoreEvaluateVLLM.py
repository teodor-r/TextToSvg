from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from pydantic import BaseModel
from openai import OpenAI
import json

def run_evaluation():
    model_name = "Qwen/Qwen3-30B-A3B"
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quantization_config,
        device_map="cuda",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    tokenizer.padding_side  = 'left'
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    eval_dataset = pd.read_csv("./data/svg_eval_dataset.csv")
    optional_prompt = "You're a chatbot for svg generation. Strictly follow the following instructions. The SVG scene should be of the following dimensions 256*256. First, think about the scene, about the objects on the stage and their characteristics. Make sure that all tags are closed and the svg is correct! Answer is only SVG without comments!"
    @torch.inference_mode
    def get_sample(row):
        user_content_prompt = f"Generate SVG image from description <{row.sentence[:-1]}>."
        return [
            {"role": "system", "content": optional_prompt},
            {"role": "user", "content": user_content_prompt}
            ]

    eval_dataset["eval_data"] = eval_dataset.apply(get_sample, axis=1)
    dataset = Dataset.from_dict({"query": list(eval_dataset["eval_data"])})
    dataset = dataset.map(lambda x: {"formatted_query": tokenizer.apply_chat_template(x["query"], tokenize=False, add_generation_prompt=True, enable_thinking=False)}, remove_columns=["query"])
    def preprocess_function(examples):
        return tokenizer(examples["formatted_query"], padding=True, truncation=False)
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=["formatted_query"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Создаем DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=data_collator,
    )
    model = torch.compile(model, backend='inductor')  
    once_checked= False
    svg_answers = []
    for i, batch in enumerate(tqdm(dataloader, desc="Генерация SVG")):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=1000
            )
            batch_output_ids = outputs[:, batch["input_ids"].size(1):]
        decoded_responses= tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
        if i % 10 == 0:
            print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        svg_answers+=decoded_responses

    cur_path = os.getcwd()
    dir_path_for_res = os.path.join(cur_path, model_name)
    print(dir_path_for_res)
    file_path = os.path.join(dir_path_for_res, "outputs_model.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp = pd.DataFrame(svg_answers)
    temp.to_csv(file_path,index = False)


def vllm_evaluation():
    model_name = "Qwen/Qwen3-30B-A3B"
    # model_name = "Qwen/Qwen3-14B"
    base_url = "http://localhost:8000/v1"
    api_key="EMPTY"
    client = OpenAI(base_url=base_url, api_key=api_key)

    eval_dataset = pd.read_csv("./data/svg_eval_dataset.csv")
    optional_prompt = "You're a chatbot for beautiful svg generation. Strictly follow the following instructions. The SVG scene should be of the following dimensions 256*256. First, think about the scene, about the objects on the stage and their characteristics. Make sure that all tags are closed and the svg is correct! Answer is only SVG without comments! Make sure that the written svg is correct!"
    @torch.inference_mode
    def get_sample(row):
        user_content_prompt = f"Generate SVG image from description: <{row.sentence[:-1]}>."
        return [
            {"role": "system", "content": optional_prompt},
            {"role": "user", "content": user_content_prompt}
            ]

    eval_dataset["eval_data"] = eval_dataset.apply(get_sample, axis=1)
    dataset = Dataset.from_dict({"query": list(eval_dataset["eval_data"])})

    svg_answers = []
    class SVG(BaseModel):
        svg: str

    json_schema = SVG.model_json_schema()
    for idx in tqdm(range(5000), desc="Генерация SVG"):
        response = client.chat.completions.create(
            model=model_name,
            messages=dataset['query'][idx],
            max_tokens=2000,
            extra_body={"guided_json": json_schema},
        )
        try:
            if idx % 100 == 0:
                print(SVG(**json.loads(response.choices[0].message.content)).svg)
                print()
            svg_answers += [SVG(**json.loads(response.choices[0].message.content)).svg]
        except Exception as e:
            print(e)
            svg_answers += ['']
            pass

    cur_path = os.getcwd()
    dir_path_for_res = os.path.join(cur_path, model_name)
    print(dir_path_for_res)
    file_path = os.path.join(dir_path_for_res, "outputs_model.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp = pd.DataFrame(svg_answers)
    temp.to_csv(file_path,index = False)

if __name__ == "__main__":
    vllm_evaluation()