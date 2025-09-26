import pandas as pd
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings

def PPG_FindFeature(av, T, weight, height, age, gender):
    # 找到所有峰值
    peaks, _ = find_peaks(av['PPG_Clean_Mean'], distance=3, prominence=0.5)
    # 找到所有谷值（反转信号找谷）
    troughs, _ = find_peaks(-av['PPG_Clean_Mean'], distance=2, prominence=0.5)

    # # 第一步：主波峰 h1: 取第一个最高的峰
    h1_index = peaks[np.argmax(av['PPG_Clean_Mean'].iloc[peaks])]
    h1_value = av['PPG_Clean_Mean'].iloc[h1_index]
    h1_time = av['Time'].iloc[h1_index]

    # # 第二步：粗略寻找降中峡
    # 计算一阶导数和二阶导数 (二阶导小于0为凹面，大于0为凸面）
    # 由于采集的脉搏波包含多种脉象，而一部分脉象的降中峡和次波峰极其不明显，所以本研究采用了结合波形一阶导和二阶导的方法，用于提取降中峡和次波峰
    first_derivative = np.gradient(av['PPG_Clean_Mean'])
    second_derivative = np.gradient(first_derivative)

    # 在主波峰之后寻找二阶导数最大的点 (搜索范围推荐：  h1_index + 5: h1_index + len(av['PPG_Clean_Mean'])//2  ）
    search_region_second = second_derivative[h1_index + 5: h1_index + len(av['PPG_Clean_Mean']) // 3]
    min_indices = np.argmax(search_region_second)  # 找到混合导数最小的两个点
    h2_index = min_indices + h1_index + 5  # 转换为全局索引
    for k1_index in range(h1_index + 5, h1_index + len(av['PPG_Clean_Mean']) // 3):
        k1 = first_derivative[k1_index]
        if k1 > 0:
            h2_index = k1_index
            # print("h2 changed")
            break
    h2_value = av['PPG_Clean_Mean'].iloc[h2_index]
    h2_time = av['Time'].iloc[h2_index]

    # # 第三步：寻找次波峰
    # 在降中峡之后寻找二阶导数最小的点 (搜索范围推荐：  h2_index + 1: h2_index + len(av['PPG_Clean_Mean'])//4  ）
    search_region_second = second_derivative[h2_index + 1: h2_index + len(av['PPG_Clean_Mean'])//4]
    min_indices = np.argmin(search_region_second)  # 找到混合导数最小的两个点
    h3_index = min_indices + h2_index + (1 + 1)   # 转换为全局索引
    for k1_index in range(h2_index + 1, h2_index + len(av['PPG_Clean_Mean'])//4):
        k1 = first_derivative[k1_index]
        k1_last = first_derivative[k1_index-1]
        if k1 < 0 and k1_last >0:
            h3_index = k1_index
            # print("h3 changed")
            break
    h3_value = av['PPG_Clean_Mean'].iloc[h3_index]
    h3_time = av['Time'].iloc[h3_index]

    plot_flag = 0    # 置1绘制平均波形图
    if plot_flag == 1:
        # 绘制平均波形图
        plt.figure(figsize=(10, 6))
        # plt.subplot(1,2,1)
        plt.plot(av["Time"], av["PPG_Clean_Mean"], label='av_pulse')
        # print('h1_time', h1_time, 'h1_value', h1_value, 'h2_time', h2_time, 'h2_value', h2_value, 'h3_time', h3_time, 'h3_value', h3_value)
        # 绘制波形和特征点
        plt.scatter(h1_time, h1_value, color='red', label='h1 (Main Peak)')
        plt.scatter(h2_time, h2_value, color='green', label='h2 (Trough)')
        plt.scatter(h3_time, h3_value, color='blue', label='h3 (Secondary Peak)')
        plt.legend()
        plt.show()

    # # 第四步：计算其他时域特征
    # 重搏波与降中峡的幅度差值
    h4 = h3_value - h2_value
    # 波形最低点(谷底)
    h5 = np.min(av['PPG_Clean_Mean'])

    # 高压持续时间(主波2/3幅值以上持续时间)
    threshold = (2 / 3) * h1_value     # 计算 2/3 幅值
    left_index = h1_index    # 向左搜索直到找到小于 threshold 的点
    while left_index > 0 and av['PPG_Clean_Mean'].iloc[left_index] >= threshold:
        left_index -= 1
    right_index = h1_index   # 向右搜索直到找到小于 threshold 的点
    while right_index < len(av) - 1 and av['PPG_Clean_Mean'].iloc[right_index] >= threshold:
        right_index += 1
    W = av['Time'].iloc[right_index] - av['Time'].iloc[left_index]

    # 脉搏周期起始点到主波波峰的斜率
    slop_rise = (h1_value - av.iloc[0]["PPG_Clean_Mean"]) / (h1_time - av.iloc[0]["Time"])
    # 主波波峰到周期结束点的斜率
    slop_fall = (av.iloc[len(av['PPG_Clean_Mean']) - 1]["PPG_Clean_Mean"] - h1_value) / (av.iloc[len(av['PPG_Clean_Mean']) - 1]["Time"] - h1_time)
    # 主波波峰到重搏波波峰的斜率
    slop_peak_diastolic = (h3_value - h1_value) / (h3_time - h1_time)

    # 各种面积
    x_values = np.array(av["Time"])
    y_values = np.array(av["PPG_Clean_Mean"])
    # 周期波形与基线之间围成的总面积
    area_single = np.trapezoid(y_values, x_values)
    # 周期起始到主波波峰与基线围成的面积
    area_start_max = np.trapezoid(y_values[0:h1_index], x_values[0:h1_index])
    # 主波波峰到降中峡与基线围成的面积
    area_max_notch = np.trapezoid(y_values[h1_index:h2_index], x_values[h1_index:h2_index])
    # 降中峡到重搏波波峰与基线围成的面积
    area_notch_diastolic = np.trapezoid(y_values[h2_index:h3_index], x_values[h2_index:h3_index])
    # 重搏波波峰到周期结束与基线围成的面积
    area_diastolic_end = np.trapezoid(y_values[h3_index:len(av['PPG_Clean_Mean']) - 1], x_values[h3_index:len(av['PPG_Clean_Mean']) - 1])

    # # 第五步：计算频域特征
    # 计算PPG 0-10hz 的频率谱能量
    fft_signal = abs(np.fft.fft(y_values))
    fft_x = np.fft.fftfreq(y_values.size, d=1 / 200)  # 频率数组
    fft_x, fft_signal = np.fft.fftshift(fft_x), np.fft.fftshift(fft_signal)  # 移位
    # plt.plot(fft_x, abs(fft_signal))
    # plt.show()
    # 利用线性插值法求被积函数
    def integrand(x):
        return np.interp(x, fft_x, fft_signal)
    # 使用 quad 函数进行数值积分
    # 禁用 IntegrationWarning
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
    # 求5个子带的频率谱能量
    freq1, error = quad(integrand, 0, 1)
    freq2, error = quad(integrand, 1, 2)
    freq3, error = quad(integrand, 2, 3)
    freq4, error = quad(integrand, 3, 4)
    freq5, error = quad(integrand, 4, 5)
    freq6, error = quad(integrand, 5, 10)

    # # 第六步：计算血流信息参数特征
    # 特征𝐾值
    Pm = np.average(av["PPG_Clean_Mean"])
    K = (Pm - h5) / (h1_value - h4)

    # 心指数 𝐶𝐼
    S1 = area_start_max + area_max_notch
    S2 = area_notch_diastolic + area_diastolic_end
    CO = 0.82 * ((1 - S1 / S2) / abs(S1 / S2))  # 心输出量 CO
    BSA = np.square((float(weight) * float(height)) / 3600)  # 体表面积
    CI = CO / BSA

    # 输出结果
    Features = pd.Series([h1_value, h2_value, h3_value,h4, h5, h1_time, h2_time-h1_time, h3_time-h2_time, T-h3_time, T, W, slop_rise, slop_fall, slop_peak_diastolic, area_single, area_start_max, area_max_notch, area_notch_diastolic, area_diastolic_end, freq1, freq2, freq3, freq4, freq5, freq6, K, CI, weight, height, age, gender],
                             ['h1', 'h2', 'h3', 'h4', 'h5', 't1', 't2-t1', 't3-t2', 'T-t3', 'T', 'W', 'slop_rise', 'slop_fall', 'slop_peak_diastolic', 'area_single', 'area_start_max', 'area_max_notch', 'area_notch_diastolic', 'area_diastolic_end', 'freq1', 'freq2', 'freq3', 'freq4', 'freq5', 'freq6', 'K', 'CI', 'weight', 'height', 'age', 'gender'])
    return Features

