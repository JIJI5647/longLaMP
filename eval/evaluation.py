import evaluate
import json
import os
from tqdm import tqdm # 强烈建议安装 tqdm: pip install tqdm
import nltk

# 加载评估插件
print("正在初始化评估指标 (可能会下载数据)...")
bleu = evaluate.load("sacrebleu")
print("1")
rouge = evaluate.load('rouge')
print("2")
meteor = evaluate.load("meteor")
print("评估指标加载完成。")


def load_results(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到结果文件: {file_path}")
    
    predictions = []
    references = []
    print(f"正在读取文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                # 过滤掉空的预测（防止计算报错）
                if record.get('prediction') and record.get('target'):
                    predictions.append(record['prediction'])
                    references.append(record['target'])
            except:
                continue
    print(f"成功加载 {len(predictions)} 条记录。")
    return predictions, references

def evaluate_generation(file_path):
    predictions, references = load_results(file_path)
    
    if not predictions:
        print("警告：数据为空，无法评估。")
        return None

    # 计算 ROUGE (对于几千条数据，这步最慢)
    print("正在计算 ROUGE 分数 (这可能需要几分钟)...")
    rouge_results = rouge.compute(predictions=predictions, references=references)
    print("ROUGE 计算完成。")
    
    # 计算 BLEU
    print("正在计算 BLEU 分数...")
    bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    print("BLEU 计算完成。")
    
    # 计算meteor 
    print("正在计算 Meteor 分数...")
    meteor_result = meteor.compute(predictions=predictions, references=references)
    print("Meteor 计算完成。")
    return {
        'rouge': rouge_results,
        'bleu': bleu_results,
        'meteor': meteor_result
    }

if __name__ == "__main__":
    # 确保路径是绝对路径或正确的相对路径
    result_file = "/root/output/val_predictions.jsonl"
    
    try:
        results = evaluate_generation(result_file)
        if results:
            print("\n" + "="*30)
            print("评估结果详情:")
            print(json.dumps(results, indent=4, ensure_ascii=False))
            print("="*30)
    except Exception as e:
        print(f"运行出错: {e}")