import json
import os
import argparse
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    pipeline
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
from data.datasets_llama import GeneralSeq2SeqDataset
from prompts.prompt import create_prompt_generator
from eval.generation_metrics import create_metric_bleu_rouge_meteor

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--model", default="google/flan-t5-base")
parser.add_argument("--dataset", default="abstract_generation_user")
parser.add_argument("--task", required=True)
parser.add_argument("--output_dir", default="/output")
parser.add_argument("--use_profile", action="store_true")
parser.add_argument("--num_support_profile", type=int, default=1)
parser.add_argument("--max_length", type = int, default = 256)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight_decay", type = float, default = 0.0001)
parser.add_argument("--num_train_epochs", type = int, default = 5)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--lr_scheduler_type", default = "linear")
parser.add_argument("--warmup_ratio", type = float, default = 0.05)
parser.add_argument("--retriever", default="bm25")
parser.add_argument("--is_ranked", action="store_true")
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
parser.add_argument("--batch_size", type=int, default=16)


if __name__ == "__main__":
    args = parser.parse_args()
    
    # 1. 加载 Tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=args.max_length)
    compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
    best_metric = "rouge-1"
    
    # 2. PEFT 配置 (LoRA)
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    print("模型和 Tokenizer 加载完成。")

    # 3. 数据集准备
    prompt_gen = None
    if args.use_profile:
        prompt_gen, _ = create_prompt_generator(
            num_retrieve=args.num_support_profile,
            tokenizer=tokenizer,
            ret_type=args.retriever,
            is_ranked=args.is_ranked,
            max_length=args.max_length
        )

    # 定义预处理逻辑
    def preprocess_fn(sample):
        model_inputs = tokenizer(sample["source"], max_length=args.max_length, truncation=True)
        labels = tokenizer(text_target=sample["target"], max_length=args.max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 加载并预处理数据集
    def get_split(split):
        ds = GeneralSeq2SeqDataset(
            dataset_name=args.dataset, task=args.task, split=split, 
            use_profile=args.use_profile, create_prompt=prompt_gen
        )
        return ds,[preprocess_fn(s) for s in ds]

    _,train_ds = get_split("train")

    raw_val, val_ds = get_split("val")
    print("数据集加载和预处理完成。")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = args.output_dir,
        do_train = True,
        do_eval = True,
        eval_strategy = "epoch",
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.lr,
        weight_decay = args.weight_decay,
        num_train_epochs = args.num_train_epochs,
        lr_scheduler_type = args.lr_scheduler_type,
        warmup_ratio = args.warmup_ratio,
        generation_num_beams = args.generation_num_beams,
        predict_with_generate = True,
        save_strategy = "epoch",
        logging_steps = 50,
        eval_accumulation_steps = 1,
        generation_max_length = args.generation_max_length,
        load_best_model_at_end = True,
        metric_for_best_model = best_metric,
        greater_is_better = True
    )

    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(os.path.join("model", args.name))
    print("模型训练完成并已保存。")

    print(trainer.evaluate(val_ds))
    '''
    model.eval()
    
    # 2. 创建 Seq2Seq 推理 Pipeline
    pipe = pipeline(
        "text2text-generation", 
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1, # 指定显卡
        batch_size=args.batch_size
    )

    # 3. 定义数据迭代器
    def prompt_iterator(dataset):
        for item in dataset:
            yield item["source"]

    # 4. 执行批量生成
    print(f"正在为 {len(val_ds)} 条数据生成预测结果...")
    outputs = pipe(
        prompt_iterator(raw_val),
        batch_size=args.batch_size,
        max_new_tokens=args.generation_max_length,
    )

    # 5. 保存结果为 JSONL
    out_path = os.path.join(args.output_dir, f"{args.name}_results.jsonl")
    
    with open(out_path, "w", encoding="utf-8") as fout:
        # zip 将预测结果、原始数据一一对应
        for out, raw_item in tqdm(zip(outputs, raw_val), total=len(raw_val)):
            # Seq2Seq pipeline 的输出直接在 'generated_text' 里
            pred = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
            
            record = {
                "input": raw_item["raw_input"],   # 对应的输入
                "target": raw_item["target"],  # 对应的金标准答案
                "prediction": pred.strip(),    # 模型预测的结果
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"结果已成功保存至: {out_path}")
    '''
    