import argparse
import os
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from data.datasets_llama import GeneralSeq2SeqDataset, prompt_iterator
from prompts.prompt import create_prompt_generator
import json
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True) # 实验名称
parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--dataset", default="abstract_generation_user") # LongLaMP数据集名称
parser.add_argument("--task", required=True) # 任务类型：如 generation_abstract
parser.add_argument("--output_dir", default="./output")
parser.add_argument("--output_dir_retrieved", required=False)
parser.add_argument("--use_profile", action="store_true") 
parser.add_argument("--retriever", default="bm25")
parser.add_argument("--num_support_profile", type=int, default=3) # 检索条数
parser.add_argument("--is_ranked", action="store_true")
parser.add_argument("--max_length", type=int, default=512)


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # 2. 初始化 RAG 提示词生成器
    prompt_gen = None
    if args.use_profile:
        prompt_gen, _ = create_prompt_generator(
            num_retrieve=args.num_support_profile,
            ret_type=args.retriever,
            is_ranked=args.is_ranked,
            max_length=args.max_length,
            tokenizer=tokenizer
        )

    # 3. 加载数据集 (使用你重构后的类，直接从 HF 下载)
    eval_dataset = GeneralSeq2SeqDataset(
        dataset_name=args.dataset,
        use_profile=args.use_profile,
        task=args.task,
        split="val", 
        create_prompt=prompt_gen
    )

    # 4. 加载模型 (4-bit 量化)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    # 5. 创建推理 Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
    )

    outputs = pipe(
        prompt_iterator(eval_dataset, tokenizer),
        batch_size=16,        
        max_new_tokens=args.max_length,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    out_path = os.path.join(args.output_dir, f"{args.name}_results.jsonl")

    # 保存结果
    with open(out_path, "w", encoding="utf-8") as fout:
        for out, raw_item in tqdm(zip(outputs, eval_dataset), total=len(eval_dataset)):
            pred = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
            
            record = {
                "input": raw_item["raw_input"],
                "target": raw_item["target"],
                "prediction": pred.strip(),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")