import argparse
import os
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.datasets_llama import load_orgin_data
import json
from transformers import pipeline
from tqdm import tqdm
from transformers import BitsAndBytesConfig

# Prevent the Transformers library from attempting online downloads. Keep this
# setting if you are working in an offline environment with local model files.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# CLI arguments are defined for integration with larger pipelines; this script
# does not currently use most of them, but they are kept for compatibility.
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--tokenizer", required=False)
parser.add_argument("--task", required=True)
parser.add_argument("--output_dir", default="./output")
parser.add_argument("--output_dir_retrieved", required=True)
parser.add_argument("--use_profile", action="store_false")
parser.add_argument("--retriever", default="bm25")
parser.add_argument("--num_support_profile", type=int, default=1)
parser.add_argument("--is_ranked", action="store_true")
parser.add_argument("--cache_dir", default="./cache")
parser.add_argument("--max_length", type=int, default=512)


if __name__ == "__main__":
    # Path to the local model snapshot. Replace with the absolute path to your
    # local checkout or cached model directory. Do not pass a remote repo id
    # here when working offline.
    model_local_path = (
        "/root/data/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    )

    print(f"Loading model from local path: {model_local_path}")

    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
    # Load tokenizer from the local directory. `local_files_only=True` avoids
    # any attempt to query the Hugging Face Hub.
    tokenizer = AutoTokenizer.from_pretrained(
        model_local_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Load dataset splits and extract validation inputs/targets
    train_ds, test_ds, val_ds = load_orgin_data()
    X_val = val_ds["input"]
    Y_val = val_ds["output"]
    print(f"Validation size: {len(X_val)}")


    # Load model. Use bfloat16 for Qwen-2.5 if your environment/GPU supports it;
    # keep device placement automatic so the library can map shards to devices.
    model = AutoModelForCausalLM.from_pretrained(
        model_local_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config, # 开启4bit
        device_map="auto",
        local_files_only=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # Prepare output directory and generation parameters
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # File to store validation predictions (JSONL): one object per line
    out_path = os.path.join(output_dir, "val_predictions.jsonl")

    # Generation parameters
    max_new_tokens = args.max_length if args.max_length is not None else 512

    # Create a text-generation pipeline that leverages model sharding and
    # internal batching. `batch_size` controls how many examples are sent to
    # the model in one pipeline call; tune this based on available GPU memory.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        batch_size=16,
    )

    # Process validation inputs in batches and write JSONL predictions.
    print("Start validation generation (batched)...")
    batch_size = 16
    with open(out_path, "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(X_val), batch_size)):
            batch_inputs = X_val[start : start + batch_size]

            # Convert messages to model input strings for each example in the
            # batch using the tokenizer's chat template helper.
            batch_texts = []
            for input_text in batch_inputs:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)

            # Run the pipeline on the batch. Use return_full_text=False so the
            # pipeline returns only the newly generated portion when possible.
            outputs = pipe(
                batch_texts, max_new_tokens=max_new_tokens, return_full_text=False
            )

            # `outputs` is a list of dicts with key 'generated_text' for each
            # input in the batch.
            for idx, out in enumerate(outputs):
                i = start + idx
                # pipeline returns dict like {'generated_text': '...'}
                pred = out.get("generated_text") if isinstance(out, dict) else str(out)

                record = {
                    "id": i,
                    "input": X_val[i],
                    "target": Y_val[i] if i < len(Y_val) else None,
                    "prediction": pred,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Progress logging
            produced = min(start + batch_size, len(X_val))
            print(f"Generated {produced}/{len(X_val)}")

    print(f"Validation predictions saved to: {out_path}")