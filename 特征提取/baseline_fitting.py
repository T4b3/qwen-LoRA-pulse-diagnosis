import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import warnings

# 该函数用于寻找每个波峰对应的波谷，据此拟合基线，修正漂移

# 输入singal为PPG_Process处理后的dataframe
def baseline_fitting(switch, signal, sampling_rate):
    # 禁用 FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    ppg_signals, info = nk.ppg_process(signal, sampling_rate)
    peaks = nk.ppg_findpeaks(ppg_signals["PPG_Clean"], sampling_rate)  # 输出峰值点，形式为几个index值的数组
    peaks = peaks['PPG_Peaks']
    origin = signal
    ppg_valleys = []

    # 寻找谷底集合(索引)
    for peak in peaks:
        if peak > 50:
            segment = origin[peak-50:peak]
        else:
            segment = origin[:peak]
        # 向前寻找一个谷底
        # valleys, _ = find_peaks(-segment, distance=5)
        # if len(valleys):
        #     ppg_valleys.append(peak - 50 + valleys[len(valleys) - 1])
        # else:
        #     ppg_valleys.append(peak - 50)
        ppg_valleys.append(segment.idxmin())

    # 根据谷底拟合基线
    ppg_valleys = np.array(ppg_valleys)
    origin_valleys = origin.iloc[ppg_valleys].values
        # 最后补一个点作为拟合基线收束
    ppg_valleys = np.append(ppg_valleys, len(origin)-1)
    origin_valleys = np.append(origin_valleys, origin_valleys[len(origin_valleys)-1])
    spline = CubicSpline(ppg_valleys, origin_valleys)
    x_fine = range(0,len(origin))
    y_fine = spline(x_fine)

    if switch:              # 开关为1，进行校正
        # 对原信号进行漂移校正
        signal = signal.copy()
        for i in x_fine:
            signal.iloc[i] = signal.iloc[i] - y_fine[i]
    else:                   # 开关为0，绘制基线
        # 绘图
        plt.figure(figsize=(12,8))
        plt.plot(origin)
        plt.scatter(ppg_valleys, origin_valleys, color='red', label='valleys')
        plt.plot(x_fine, y_fine, label='baseline')

    # 恢复警告设置
    warnings.resetwarnings()

    return signal