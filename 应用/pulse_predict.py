import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def make_prompt(channel1, channel2):
    # return f"食指脉搏波特征：{channel1}；中指脉搏波特征：{channel2}；根据脉搏特征判断脉象类别：弦脉、沉脉、迟脉、虚脉、浮脉、细脉？你必须回答上述脉象中的一个，不允许输出其他内容，答案："
    return (
        f"已给出两路脉搏波特征，请在[沉脉,迟脉,虚脉,弦脉,浮脉,细脉]中选择最可能的一类。你必须回答上述脉象中的一个，不允许输出其他内容.\n通道一: {channel1}\n通道二: {channel2}\n答案："
    )

def load_model(model_path, adapter_path):
    # 加载tokenizer
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("正在加载模型...")
    try:
        # 首先尝试直接加载合并后的模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        # 如果需要加载额外的LoRA权重
        if adapter_path and adapter_path != model_path:
            print("正在加载额外的LoRA权重...")
            model = PeftModel.from_pretrained(
                model, 
                adapter_path,
                is_trainable=False  # 设置为推理模式
            )
    except Exception as e:
        print(f"模型加载出错: {str(e)}")
        raise

    model.eval()
    return tokenizer, model

def normalize_label(text):
    """提取文本中的"X脉"标签"""
    # 所有可能的脉象类别
    VALID_LABELS = ["沉脉", "迟脉", "虚脉", "弦脉", "浮脉", "细脉"]
    
    # 遍历寻找任意一个有效标签
    for label in VALID_LABELS:
        if label in text:
            return label
    return text.strip()  # 如果没找到，返回原文本

def predict(json_path, model_path, adapter_path="./qwen-pulse-lora"):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载模型和tokenizer
    tokenizer, model = load_model(model_path, adapter_path)
    
    # 构建prompt
    prompt = make_prompt(data['channel1'], data['channel2'])
    print(f"\n输入prompt:\n{prompt}")
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            num_return_sequences=1,
            temperature=0.1,
            top_p=0.95,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    normalized_response = normalize_label(response)
    print(f"\n模型原始输出: {response}")
    print(f"提取脉象类别: {normalized_response}")
    
    # 对比真实标签
    if 'label' in data:
        true_label = normalize_label(data['label'])
        is_correct = normalized_response == true_label
        print(f"真实标签: {true_label}")
        print(f"预测正确: {'✓' if is_correct else '✗'}")
    
    return normalized_response

if __name__ == "__main__":
    # 配置路径
    json_path = r"E:\T4b3_Works\AnacondaWorks\data1.8\feature_json\sample_00368.json"
    
    # 修改为本地模型路径
    model_path = r"E:\T4b3_Works\AnacondaWorks\qwen_train\qwen-pulse-lora"  # 本地模型路径
    adapter_path = r"E:\T4b3_Works\AnacondaWorks\qwen_train\qwen-pulse-lora"  # 本地LoRA权重路径
    
    # 执行预测
    predict(json_path, model_path, adapter_path)
