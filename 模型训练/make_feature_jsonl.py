import os
import json
from tqdm import tqdm

# ====== I/O 路径 ======
json_dir  = r"E:\T4b3_Works\AnacondaWorks\data1.8\feature_json"           # 输入：你的 sample_*.json 所在文件夹
out_jsonl = r"E:\T4b3_Works\AnacondaWorks\data1.8\feature_json\jsonl\train.jsonl"  # 输出：通用SFT jsonl

os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

# ====== 标签规范化 ======
LABEL_CANON = {
    "沉": "沉脉", "沉脉": "沉脉",
    "迟": "迟脉", "迟脉": "迟脉",
    "虚": "虚脉", "虚脉": "虚脉",
    "弦": "弦脉", "弦脉": "弦脉",
    "浮": "浮脉", "浮脉": "浮脉",
    "细": "细脉", "细脉": "细脉",
}
LABEL_SET = set(LABEL_CANON.values())

def normalize_label(y: str) -> str:
    y = str(y).strip()
    return LABEL_CANON.get(y, y)

# ====== 工具：把数组或其它类型统一成逗号串 ======
def to_comma_string(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        return ",".join(str(v) for v in x)
    return str(x)

# ====== 组装 prompt ======
def make_prompt(ch1_str: str, ch2_str: str) -> str:
    # 语言尽量稳定、短小，利于 token 压缩
    return (
        "已给出两路脉搏波特征，请在{沉脉,迟脉,虚脉,弦脉,浮脉,细脉}中选择最可能的一类。"
        f"\n通道一: {ch1_str}\n通道二: {ch2_str}\n答案："
    )
    # return f"食指脉搏波特征：{ch1_str}；中指脉搏波特征：{ch2_str}；请判断脉象类别：沉脉、迟脉、实脉、弦脉、滑脉、细脉？你只能回答上述脉象中的一个，不需要其他解释，答案："


files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
print(f"将处理 {len(files)} 个文件...")

kept, skipped = 0, 0
with open(out_jsonl, "w", encoding="utf-8") as fout:
    for fname in tqdm(files):
        fpath = os.path.join(json_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fin:
                obj = json.load(fin)
        except Exception:
            skipped += 1
            continue

        raw_label = obj.get("label", "")
        ch1_str = to_comma_string(obj.get("channel1"))
        ch2_str = to_comma_string(obj.get("channel2"))

        label = normalize_label(raw_label)
        if label not in LABEL_SET:
            skipped += 1
            continue

        prompt = make_prompt(ch1_str, ch2_str)
        line = {"prompt": prompt, "answer": label}

        fout.write(json.dumps(line, ensure_ascii=False) + "\n")
        kept += 1

print(f"✅ 已生成：{out_jsonl} | 保留 {kept} 条，跳过 {skipped} 条")
