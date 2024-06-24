import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import scipy.signal


# Cos signal properties for 1 task
N_cos = 1000
sample_rate_cos = 10000
freq_1, freq_2, freq_3 = 50, 150, 450
amplitude_1, amplitude_2, amplitude_3 = 1, 1, 1


def f_cos(x):
    return amplitude_1 * np.cos(freq_1 * x * 2 * np.pi) + \
           amplitude_2 * np.cos(freq_2 * x * 2 * np.pi) + amplitude_3 * np.cos(freq_3 * x * 2 * np.pi)


def init_cos_arr():
    signal_ = list()
    time_arr = list()
    for i in range(N_cos):
        signal_.append(f_cos(i / sample_rate_cos))
        time_arr.append(i / sample_rate_cos)
    return signal_, time_arr


def cos_50():
    signal_ = list()
    for i in range(N_cos):
        signal_.append(np.cos(i / sample_rate_cos * 2 * np.pi * freq_1))
    return signal_


def cos_150():
    signal_ = list()
    for i in range(N_cos):
        signal_.append(np.cos(i / sample_rate_cos * 2 * np.pi * freq_2))
    return signal_


def cos_450():
    signal_ = list()
    for i in range(N_cos):
        signal_.append(np.cos(i / sample_rate_cos * 2 * np.pi * freq_3))
    return signal_


def cos_50_450():
    signal_ = list()
    for i in range(N_cos):
        signal_.append(np.cos(i / sample_rate_cos * 2 * np.pi * freq_3) + np.cos(i / sample_rate_cos * 2 * np.pi * freq_2))
    return signal_


signal, time_arr = init_cos_arr()
signal_50 = cos_50()
signal_150 = cos_150()
signal_450 = cos_450()
signal_no_150 = cos_50_450()
freq_axis = 2 * np.pi * np.linspace(0, sample_rate_cos, N_cos)

fig, axs = plt.subplots(2)
axs[0].set_title("Signal")
axs[0].plot(time_arr, signal)
axs[1].set_title("Fourier")
axs[1].plot(scipy.fft.fftfreq(N_cos)[:N_cos // 2], np.abs(scipy.fft.fft(signal))[:N_cos // 2])
plt.show()


def LFF(cut_of_freq, sample_rate, N):
    wc = 2 * np.pi * cut_of_freq
    freq_axis = 2 * np.pi * np.linspace(0, sample_rate, N)
    halfLowFreqFilter = wc ** 2 / (- freq_axis[:N // 2] ** 2
                                   + 1j * wc * freq_axis[:N // 2] * np.sqrt(2) + wc ** 2)
    return np.concatenate((halfLowFreqFilter, np.flip(halfLowFreqFilter)))


def HFF(cut_of_freq, sample_rate, N):
    wc = 2 * np.pi * cut_of_freq
    freq_axis = 2 * np.pi * np.linspace(0, sample_rate, N)
    halfHighFreqFilter = freq_axis[:N // 2] ** 2 / (- freq_axis[:N // 2] ** 2
                                                    + 1j * wc * freq_axis[:N // 2] * np.sqrt(2) + wc ** 2)
    return np.concatenate((halfHighFreqFilter, np.flip(halfHighFreqFilter)))


def Order_Filter(order, wc, w):
    p = np.zeros(2*order, dtype=complex)
    for k in range(order):
        theta = np.pi/2 + (2*k + 1)*np.pi/(2*order)
        p[2*k] = -np.sin(theta) * np.exp(1j*np.cos(theta))
        p[2*k+1] = -np.sin(theta) * np.exp(-1j*np.cos(theta))
    s = 1j * w[:len(w)//2]/wc
    result = np.arange(len(w)//2, dtype=complex)
    for k in range(2 * order):
        result = result/(s - p[k])
    return np.concatenate((result, np.flip(result)))


freq_axis = np.linspace(0, sample_rate_cos, N_cos)
signal_sp = scipy.fft.fft(signal)
filtered_sp = LFF(100, sample_rate_cos, N_cos) * signal_sp
filtered_signal = np.real(scipy.fft.ifft(filtered_sp))

plt.title("LFF")

plt.plot(time_arr, signal)
plt.plot(time_arr, signal_50)
plt.plot(time_arr, filtered_signal, color='r')
plt.legend(["Signal", "Clear signal", "Filtered Signal"])
plt.show()

pass_filter = LFF(400, sample_rate_cos, N_cos) * HFF(60, sample_rate_cos, N_cos)
stop_filter = LFF(60, sample_rate_cos, N_cos) + HFF(400, sample_rate_cos, N_cos)

fig, axs = plt.subplots(2)
axs[0].set_title("Pass Filter")
axs[0].plot(freq_axis, np.abs(pass_filter))
axs[1].set_title("Stop Filter")
axs[1].plot(freq_axis, np.abs(stop_filter))
plt.show()

signal_sp = scipy.fft.fft(signal)
filtered_sp = pass_filter * signal_sp
filtered_signal = np.real(scipy.fft.ifft(filtered_sp))

signal_150_sp = scipy.fft.fft(signal_150)

fig, axs = plt.subplots(nrows=2, ncols=3)
plt.title("Pass Filter")
axs[0][0].plot(time_arr, signal)
axs[0][1].plot(time_arr, signal_150)
axs[0][2].plot(time_arr, filtered_signal)
axs[1][0].plot(freq_axis, np.abs(signal_sp[:N_cos]))
axs[1][1].plot(freq_axis, np.abs(signal_150_sp[:N_cos]))
axs[1][2].plot(freq_axis, np.abs(filtered_sp[:N_cos]))
plt.show()

plt.plot(time_arr, signal)
plt.plot(time_arr, signal_150)
plt.plot(time_arr, filtered_signal, color='r')
plt.legend(["Signal", "Clear signal", "Filtered Signal"])
plt.show()

signal_sp = scipy.fft.fft(signal)
filtered_sp = stop_filter * signal_sp
filtered_signal = np.real(scipy.fft.ifft(filtered_sp))

fig, axs = plt.subplots(nrows=2, ncols=3)
plt.title("Stop Filter")
axs[0][0].plot(time_arr, signal)
axs[0][1].plot(time_arr, signal_150)
axs[0][2].plot(time_arr, filtered_signal)
axs[1][0].plot(freq_axis, np.abs(signal_sp[:N_cos]))
axs[1][1].plot(freq_axis, np.abs(signal_150_sp[:N_cos]))
axs[1][2].plot(freq_axis, np.abs(filtered_sp[:N_cos]))
plt.show()

plt.plot(time_arr, signal)
plt.plot(time_arr, signal_no_150)
plt.plot(time_arr, filtered_signal, color='r')
plt.legend(["Signal", "Clear signal", "Filtered Signal"])
plt.show()

signal_sp = scipy.fft.fft(signal)
filtered_sp = Order_Filter(4, 2 * np.pi * 120, 2 * np.pi * freq_axis) * signal_sp
filtered_signal = np.real(scipy.fft.ifft(filtered_sp))

signal_sp_scipy = scipy.fft.fft(signal)
sos = scipy.signal.butter(4, 2 * np.pi * 120, 'hp', fs=2000, output='sos')
filtered_sp_scipy = scipy.signal.sosfilt(sos, signal)
filtered_signal_scipy = np.real(scipy.fft.ifft(filtered_sp_scipy))

fig, axs = plt.subplots(nrows=2, ncols=2)
plt.title("Order Filter")
axs[0][0].plot(time_arr, signal)
axs[0][1].plot(time_arr, signal_150)
axs[1][0].plot(freq_axis, np.abs(signal_sp[:N_cos]))
axs[1][1].plot(freq_axis, np.abs(signal_150_sp[:N_cos]))
plt.show()