import os
import json
import pandas as pd
from Feature_new import PPG_FindFeature
from baseline_fitting import baseline_fitting
import neurokit2 as nk

# ================== 工具函数 ==================
def preprocess_ppg(ppg_data):
    """基线拟合 + nk 清洗"""
    ppg_data = baseline_fitting(1, ppg_data, 200)
    ppg_signals, info = nk.ppg_process(ppg_data, sampling_rate=200)
    return ppg_signals["PPG_Clean"].tolist()

def read_patient_info(txt_path):
    """读取病人信息 + 脉象标签(Pause)"""
    encodings = ["ansi", "utf-8", "gbk"]
    content = None
    for enc in encodings:
        try:
            with open(txt_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        raise UnicodeDecodeError("无法识别的编码，请检查文件：", txt_path, 0, 0, "unknown encoding")

    info = {"gender": None, "height": None, "weight": None, "age": None, "label": None}
    for line in content.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = [x.strip() for x in line.split(":", 1)]
        kl = k.lower()
        if kl.startswith("gender") or "性别" in k:
            info["gender"] = v
        elif kl.startswith("height") or "身高" in k:
            try: info["height"] = float(v)
            except: info["height"] = None
        elif kl.startswith("weight") or "体重" in k:
            try: info["weight"] = float(v)
            except: info["weight"] = None
        elif kl.startswith("age") or "年龄" in k:
            try: info["age"] = int(float(v))
            except: info["age"] = None
        elif kl.startswith("pause") or "脉象" in k:
            info["label"] = v.strip()
    return info

def feats_to_str(feats, sig=6):
    """Series / list → 用逗号拼接的字符串，控制有效数字"""
    try:
        it = feats.tolist() if hasattr(feats, "tolist") else list(feats)
    except Exception:
        it = feats
    return ",".join(f"{float(x):.{sig}g}" for x in it)

LABEL_CANON = {
    "沉": "沉脉", "沉脉": "沉脉",
    "迟": "迟脉", "迟脉": "迟脉",
    "虚": "虚脉", "虚脉": "虚脉",
    "弦": "弦脉", "弦脉": "弦脉",
    "浮": "浮脉", "浮脉": "浮脉",
    "细": "细脉", "细脉": "细脉",
}
def normalize_label(y):
    y = str(y).strip()
    return LABEL_CANON.get(y, y)

def encode_gender(g):
    """女→0, 男→1；未知回退到1（你可改成 None）"""
    if g is None:
        return 1
    s = str(g).strip()
    if s in ("女", "female", "Female", "F", "f"):
        return 0
    return 1

# ================== 主程序 ==================
base_dir = r"E:\T4b3_Works\AnacondaWorks\data1.8\evaluate_csv"
output_dir = os.path.join(base_dir, "feature_json")
os.makedirs(output_dir, exist_ok=True)

all_files = sorted(os.listdir(base_dir))
triplets = []

# 每3个一组（csv1, csv2, txt）
for i in range(0, len(all_files), 3):
    group = all_files[i:i+3]
    if len(group) == 3 and group[0].endswith(".csv") and group[1].endswith(".csv") and group[2].endswith(".txt"):
        triplets.append(group)
    else:
        print(f"[跳过] 第{i//3 + 1}组文件格式不符：{group}")

sample_id = 0

for idx, (csv1, csv2, info_txt) in enumerate(triplets):
    path_csv1 = os.path.join(base_dir, csv1)
    path_csv2 = os.path.join(base_dir, csv2)
    path_txt  = os.path.join(base_dir, info_txt)

    # 读取患者信息 & 标签
    patient_information = read_patient_info(path_txt)
    raw_label = patient_information.get("label")
    if not raw_label:
        print(f"[跳过] 无法从 {info_txt} 提取脉象标签(Pause)")
        continue
    label = normalize_label(raw_label)

    # 读取csv数据（表头列名为 PPGdata）
    df1 = pd.read_csv(path_csv1)
    df2 = pd.read_csv(path_csv2)

    if len(df1) < 9000 or len(df2) < 9000:
        print(f"[跳过] 第{idx+1}组数据不足9000行：{csv1}, {csv2}")
        continue

    # 预处理
    ppg1 = preprocess_ppg(df1["PPGdata"])
    ppg2 = preprocess_ppg(df2["PPGdata"])

    # 分三段，每段 3000 点
    for seg in range(3):
        start, end = seg * 3000, seg * 3000 + 3000
        epoch_ppg1 = ppg1[start:end]
        epoch_ppg2 = ppg2[start:end]
        if len(epoch_ppg1) < 100 or len(epoch_ppg2) < 100:
            continue

        # 处理与心动周期
        ppg_signals1, _ = nk.ppg_process(epoch_ppg1, sampling_rate=200)
        ppg_signals2, _ = nk.ppg_process(epoch_ppg2, sampling_rate=200)
        peaks1 = nk.ppg_findpeaks(ppg_signals1["PPG_Clean"], sampling_rate=200)
        peaks2 = nk.ppg_findpeaks(ppg_signals2["PPG_Clean"], sampling_rate=200)

        if len(peaks1["PPG_Peaks"]) == 0 or len(peaks2["PPG_Peaks"]) == 0:
            # 无峰值，跳过该段
            continue

        T1 = len(epoch_ppg1) / 200 / max(1, len(peaks1["PPG_Peaks"]))
        T2 = len(epoch_ppg2) / 200 / max(1, len(peaks2["PPG_Peaks"]))

        start_time = -T1 * 0.2
        end_time   =  T1 * 0.8
        hb1 = nk.epochs_create(ppg_signals1, events=peaks1["PPG_Peaks"], sampling_rate=200,
                               epochs_start=start_time, epochs_end=end_time)
        start_time = -T2 * 0.2
        end_time   =  T2 * 0.8
        hb2 = nk.epochs_create(ppg_signals2, events=peaks2["PPG_Peaks"], sampling_rate=200,
                               epochs_start=start_time, epochs_end=end_time)

        epochs_av1 = nk.epochs_average(hb1, which=["PPG_Clean"], show=False)
        epochs_av2 = nk.epochs_average(hb2, which=["PPG_Clean"], show=False)

        # ========== 调用 Feature_new，返回 pandas.Series ==========
        gender_code = encode_gender(patient_information.get("gender"))
        features1 = PPG_FindFeature(
            epochs_av1, T1,
            patient_information.get("weight"),
            patient_information.get("height"),
            patient_information.get("age"),
            gender_code,
        )
        features2 = PPG_FindFeature(
            epochs_av2, T2,
            patient_information.get("weight"),
            patient_information.get("height"),
            patient_information.get("age"),
            gender_code,
        )

        # 文本化（逗号分隔），满足 sample 结构
        ch1_str = feats_to_str(features1, sig=6)
        ch2_str = feats_to_str(features2, sig=6)

        sample = {
            "label": label,
            "channel1": ch1_str,
            "channel2": ch2_str
        }

        # 保存为 JSON
        output_path = os.path.join(output_dir, f"sample_{sample_id:05}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False)
        sample_id += 1

print(f"\n✅ 处理完成，共生成 {sample_id} 个样本，保存至：{output_dir}")
