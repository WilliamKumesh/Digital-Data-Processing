import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy import interpolate

# 1 Task


def GetRightLEftBorder(pos, k, max_length):
    return int(max(0, pos-k)), int(min(max_length, pos+k))


def SignalAverage(signal, k):
    N = len(signal)
    y = np.zeros(N)
    for i in range(N):
        l, r = GetRightLEftBorder(i, k, N)
        for j in range(l, r):
            y[i] += signal[j]
        y[i] = y[i] / (2 * k + 1)
    return y


n = 5000
p = 15
noise_amp = 5
ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p)*30)
noise = noise_amp * np.random.randn(n)
signal1 = ampl + noise
# subtract mean to eliminate DC
signal1 = signal1 - np.mean(signal1)

signal_filter = SignalAverage(signal1, 10)
plt.title("Simple Filtered Noise Signal")
plt.plot(signal1)
plt.plot(signal_filter)
plt.show()

# 2 Task


def GaussAverage(signal, k, w):
    N = len(signal)
    y = np.zeros(N)

    for i in range(N):
        weights = np.exp(-4 * np.log(2) * (np.arange(N)-i)**2/w**2)
        y[i] = np.sum(weights * signal) / np.sum(weights)
        # l, r = GetRightLEftBorder(i, k, N)
        # g = np.exp(-(4 * np.log(2) * (np.arange(r-l)) ** 2) / (w ** 2))
        # y[i] = np.sum(signal[l:r] * g) / np.sum(g)
    return y


signal_filter = GaussAverage(signal1, 10, 10)

plt.title("Gauss Filtered Noise Signal")
plt.plot(signal1)
plt.plot(signal_filter)
plt.show()

# 3 Task

import random


def SplashSignal(n, num_ones, ampl):
    A = ampl
    arr = np.zeros(n)
    indices = np.random.choice(n, size=num_ones, replace=False)
    if ampl == 0:
        arr[indices] = random.sample(range(0, 500), num_ones)
        return arr
    arr[indices] = A
    return arr


splash_signal = SplashSignal(1000, 1000//15, 1)
signal_filter = GaussAverage(splash_signal, 5, 15)

plt.title("Gauss Filtered Splash Signal")
plt.plot(splash_signal)
plt.plot(signal_filter)
plt.show()

# Task 4


def MedianFilter(signal, k, max_value):
    N = len(signal)
    arr = np.zeros(N)
    for i in range(N):
        l, r = GetRightLEftBorder(i, k, N)
        arr[i] = np.sum(signal[l:r][signal[l:r] <= max_value])/(r-l)
    return arr


filter_signal = MedianFilter(signal1, 10, 100)

splash_signal = SplashSignal(1000, 1000//10, 0)

filter_signal_splash = MedianFilter(splash_signal, 10, 300)

plt.title("Median Filtered Noise Signal")
plt.plot(signal1)
plt.plot(filter_signal)
plt.show()

plt.title("Median Filtered Splash Signal")
plt.plot(splash_signal)
plt.plot(filter_signal_splash)
plt.show()
#
# # Task 5
#
#

import numpy as np
from scipy.stats import linregress


def RemoveLinearTrend(signal):
    n = len(signal)
    bic_original = np.inf

    for i in range(1, n + 1):
        x = np.arange(n)
        slope, intercept, _, _, _ = linregress(x, signal)
        fitted = slope * x + intercept

        residuals = signal - fitted
        rss = np.sum(residuals ** 2)

        # Вычисление критерия Байесовской информации (BIC)
        bic = n * np.log(rss / n) + i * np.log(n)

        if bic < bic_original:
            bic_original = bic
            best_fit = fitted

    detremened_signal = signal - best_fit
    return detremened_signal


def CreateLinearTrendSignal(N, max_value):
    step = max_value / N
    signal_ = np.zeros(N)
    for i in range(N):
        coeff = 1
        if i > N//2:
            coeff = 0.9
        signal_[i] = i * step * coeff + random.randint(0, max_value//10)
    return signal_


linear_trend = CreateLinearTrendSignal(1000, 500)
remove_linear = RemoveLinearTrend(linear_trend)
plt.plot(linear_trend, label='LinearTrend')
plt.plot(remove_linear, label='NoLinearTrend')
plt.legend()
plt.show()

# Task 6

N = 10000
time = np.arange(0, N)/N
signal2 = np.exp(.5*np.random.randn(N))
n_outliers = 50
rand_points = np.random.randint(0, N, n_outliers)
signal2[rand_points] = np.abs(np.random.randn(n_outliers) * (np.max(signal2) - np.min(signal2)) * 10)


def RemoveOutliers(signal, threshold=3):
    mean = np.mean(signal)
    std = np.std(signal)

    filtered_signal = [x for x in signal if (x - mean) < threshold * std]

    return np.array(filtered_signal)


plt.plot(time, signal2, 'ks-')
plt.show()

filter_signal = RemoveOutliers(signal2, 6)
plt.plot(filter_signal, 'ks-')
plt.show()


# Task 7

n = 2000
p = 15
signal3 = np.interp(np.linspace(0,p,n), np.arange(0,p),np.random.rand(p)*30)
signal3 = signal3 + np.random.randn(n)
# 200:221
signal3[200:221] = signal3[200:221] + np.random.randn(21) * 9
signal3[1500:1601] = signal3[1500:1601] + np.random.randn(101) * 9

plt.plot(signal3)
plt.show()

pct_win = 2
k = int(n*(pct_win/2/100))
rms_ts = np.zeros(n)
for i in range(0, n):
    l, r = GetRightLEftBorder(i, k, n)
    temp_sig = signal3[l:r]
    temp_sig = temp_sig - np.mean(temp_sig)
    rms_ts[i] = np.sqrt(np.sum(temp_sig**2))

plt.plot(rms_ts, 's-', label='local RMS')
plt.show()

thresh = 20

signalR = signal3.copy()
signalR[rms_ts > thresh] = np.NaN
thresh_arr = signal3.copy()
thresh_arr[rms_ts < thresh] = np.NaN

fig, axs = plt.subplots(2)
axs[0].plot(signalR)
axs[1].plot(thresh_arr)
axs[1].set_xlim(-100, n + 100)
plt.show()

# 8 Task


def SpectralInterpolation(signal):
    start_nun_pos = 0
    last_nun_pos = 0
    full_signal = signal.copy()

    for i in range(len(signal) - 1):
        if np.isnan(signal[i]) and np.isnan(signal[i + 1]) and start_nun_pos == 0:
            start_nun_pos = i - 1
        if np.isnan(signal[i]) and not np.isnan(signal[i + 1]):
            last_nun_pos = i + 1
        if last_nun_pos != 0 and start_nun_pos != 0:

            window = np.arange(start_nun_pos, last_nun_pos)
            window_size = len(window)

            fft_previous = fft(full_signal[start_nun_pos-window_size:start_nun_pos])
            fft_next = fft(full_signal[last_nun_pos:last_nun_pos + window_size])
            fft_result = (fft_previous + fft_next) / 2

            regular_signal = ifft(fft_result)
            average_point = (signal[start_nun_pos - 1] + signal[last_nun_pos + 1]) / 2
            if abs(signal[start_nun_pos] - regular_signal[0]) > 15:
                regular_signal = RemoveLinearTrend(regular_signal)
                regular_signal += average_point

            for j in range(len(regular_signal)):
                full_signal[j + start_nun_pos] = regular_signal[j]


            start_nun_pos = 0
            last_nun_pos = 0

    return full_signal


signal_regular = SpectralInterpolation(signalR.copy())

fig, axs = plt.subplots(2)
axs[0].plot(signalR)
axs[1].plot(signal_regular)
plt.show()
