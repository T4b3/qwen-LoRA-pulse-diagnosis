from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from peft import TaskType
import re
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import torch


###### 1. 加载 Qwen 预训练模型 + LoRA 配置  ######
print("开始加载模型和tokenizer...")
model_name = "Qwen/Qwen-7B-Chat"  # 可以替换为你本地模型或官方路径
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,        # 或 bfloat16，按显卡支持
    low_cpu_mem_usage=True
)
# pad_token 建议还是补上（否则后面做动态 padding 也可能需要）
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

print("配置LoRA参数...")
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_proj"],
)
model = get_peft_model(model, lora_config)
print("✅ LoRA配置完成")
# 降显存
model.config.use_cache = False               # 训练时必须关，否则显存炸
model.gradient_checkpointing_enable()        # 降激活显存
model.enable_input_require_grads()           # PEFT + checkpointing 需要

###### 2. 加载数据集与 Tokenizer 转换 ######
print("\n开始加载训练数据...")
train_jsonl = r"E:/T4b3_Works/AnacondaWorks/data1.8/feature_json/jsonl/train.jsonl"
train_data = load_dataset('json', data_files=train_jsonl)['train']
print(f"✅ 成功加载训练数据，数据集大小: {len(train_data)}条")
# 如有 val.jsonl
# val_data = load_dataset('json', data_files='val.jsonl')['train']
val_data = None  # 没有验证集时可以设为None

def preprocess(example):
    # 拼接prompt和答案，适配大模型监督训练格式
    text = example['prompt'] + example['answer']
    enc = tokenizer(text, truncation=True, max_length=256)  # 先 256
    enc["labels"] = enc["input_ids"].copy()
    return enc
# def preprocess(example):
#     prompt = example["prompt"]
#     answer = example["answer"]

#     # 在答案后补 EOS
#     full_text = prompt + answer

#     enc = tokenizer(
#         full_text,
#         truncation=True,
#         max_length=256,#512   # 允许稍微长一点，特征串挺长的话 512 更安全
#         padding=False
#     )

#     # 构造 labels，并 mask 掉 prompt 部分
#     labels = enc["input_ids"].copy()

#     # 重新tokenize拿到 prompt 的 token 长度（与 truncation 一致）
#     prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
#     prompt_len = len(prompt_ids)
#     if prompt_len > len(labels):
#         prompt_len = len(labels)  # 极端截断情况保护

#     # mask 掉 prompt 区间
#     labels[:prompt_len] = [-100] * prompt_len
#     enc["labels"] = labels
#     return enc

print("\n开始数据预处理...")
train_data = train_data.map(preprocess, batched=False)
print("✅ 训练数据预处理完成")
if val_data is not None:
    val_data = val_data.map(preprocess, batched=False)
    print("✅ 验证数据预处理完成")

# 5) data collator  ??
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


###### 3. 配置训练参数并训练模型 ######
print("\n开始配置训练参数...")
training_args = TrainingArguments(
    output_dir="./qwen-pulse-lora",
    per_device_train_batch_size=1,     # 显存够可以调大
    gradient_accumulation_steps=8,           # 8 步累积 = 等效 bs=8
    bf16=True,                               # 4080 也可用 bf16=True（二选一）
    # per_device_eval_batch_size=2,
    num_train_epochs=5,
    learning_rate=2e-5,
    # learning_rate=5e-4,                      # LoRA 可适当大一点
    evaluation_strategy="epoch" if val_data is not None else "no",
    # save_strategy="epoch",
    save_strategy="steps",
    save_steps=100,                         # 每100步保存一次
    # fp16=True,                         # 显存紧张时建议开启
    logging_steps=10
)
print("✅ 训练参数配置完成")

print("\n开始训练模型...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.train()

###### 4. 保存模型与 Tokenizer ######
print("\n正在保存模型和tokenizer...")
model.save_pretrained("./qwen-pulse-lora")
tokenizer.save_pretrained("./qwen-pulse-lora")
print("✅ 训练结束，模型和tokenizer已保存！")