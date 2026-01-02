import evaluate

# 加载评估插件
rouge = evaluate.load("rouge")

predictions = ["模型生成的答案1", "模型生成的答案2"]
references = ["标准答案1", "标准答案2"]

results = rouge.compute(predictions=predictions, references=references)

print(results)