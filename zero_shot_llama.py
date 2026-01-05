import argparse
import os
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from data.datasets_llama import GeneralSeq2SeqDataset
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
    # 注意：这里我们加载测试集 split="test"
    eval_dataset = GeneralSeq2SeqDataset(
        dataset_name=args.dataset,
        use_profile=args.use_profile,
        task=args.task,
        split="test", 
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
        batch_size=16, # 根据显存调整
    )

    out_path = os.path.join(args.output_dir, f"{args.name}_predictions.jsonl")
    
    print(f"开始推理任务: {args.task}，样本总数: {len(eval_dataset)}")
    
    with open(out_path, "w", encoding="utf-8") as fout:
        # 使用 DataLoader 思想进行 Batch 处理
        batch_size = 16
        for i in tqdm(range(0, len(eval_dataset), batch_size)):
            batch_range = range(i, min(i + batch_size, len(eval_dataset)))
            
            batch_texts = []
            batch_raw_data = []
            
            for idx in batch_range:
                data_item = eval_dataset[idx] # 这里会自动触发 create_prompt
                source_text = data_item["source"]
                
                # 应用 Llama-3 聊天模板
                messages = [
                    {"role": "system", "content": "You are a personalized assistant."},
                    {"role": "user", "content": source_text},
                ]
                input_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(input_prompt)
                batch_raw_data.append(data_item)

            # 执行推理
            outputs = pipe(
                batch_texts,
                max_new_tokens=args.max_length,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )

            # 保存结果
            for idx, out in enumerate(outputs):
                if isinstance(out, list):
                    pred = out[0]["generated_text"]
                else:
                    pred = out["generated_text"]
                record = {
                    "input": batch_raw_data[idx]["source"],
                    "target": batch_raw_data[idx]["target"],
                    "prediction": pred.strip(),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"推理完成，结果已保存至: {out_path}")