import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

# 1 Task

# Cos signal properties for 1 task
N_cos = 1024
iter_delimiter_cos = 10000
freq_1 = 50
freq_2 = 150
amplitude = 1


def f_cos(x):
    return amplitude*np.cos(freq_1 * x * 2 * np.pi) + amplitude*np.cos(freq_2 * x * 2 * np.pi)


def DFT_slow(signal_func, N, iter_delimiter):

    signal_ = list()
    time_arr = list()

    for i in range(N):
        iter = i/iter_delimiter
        signal_.append(signal_func(iter))
        time_arr.append(iter)

    n = np.arange(N)
    k = np.reshape(n, (N, 1))

    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, signal_)

    return X, signal_, time_arr


# 1 - a
X, signal_cos, time_arr = DFT_slow(f_cos, N_cos, iter_delimiter_cos)

fourier = scipy.fft.fft(signal_cos)

freq_arr = scipy.fft.fftfreq(len(time_arr), freq_1)

plt.title("Cos Signal")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.plot(time_arr, signal_cos)
plt.show()

plt.title("Fourier scipy")
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.plot(freq_arr, np.abs(fourier))
plt.show()

plt.title("DFT_slow")
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.plot(freq_arr, np.abs(X))
plt.show()

# 1 - b

signal_after_fourier = scipy.fft.ifft(X)

plt.title("ifft our signal")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.plot(time_arr,  signal_after_fourier)
plt.show()


# 1 - c

def f_cos_noise(x):
    return amplitude*np.cos(freq_1 * x * 2 * np.pi) + np.random.random(1)[0] / 2 + amplitude*np.cos(freq_2 * x * 2 * np.pi) + np.random.random(1)[0] / 2


X, signal, time_arr = DFT_slow(f_cos_noise, N_cos, iter_delimiter_cos)
signal_after_fourier = scipy.fft.ifft(X)

plt.title("ifft our noise signal")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.plot(time_arr,  signal_after_fourier)
plt.show()

# 2 Task

# Rectangle signal properties
amplitude_rect = 2
T = 2
N = 400
iter_delimiter = 100


def f_rect(x):
    if x % T < 0.5 * T:
        return amplitude_rect
    return -amplitude_rect


def f_rect_noise(x):
    if x % T >= 0.5 * T:
        return amplitude + np.random.random(1)[0] / 4
    return -amplitude - np.random.random(1)[0] / 4


X, signal, time_arr = DFT_slow(f_rect, N, iter_delimiter)
fourier = scipy.fft.fft(signal)

plt.title("Signal rect")
plt.plot(time_arr, signal)
plt.show()

freq_arr = scipy.fft.fftfreq(len(signal))

plt.title("DFT_slow rect")
plt.plot(freq_arr, np.abs(X))
plt.show()

plt.title("fft")
plt.plot(freq_arr, np.abs(fourier))
plt.show()

X, signal, time_arr = DFT_slow(f_rect_noise, N, iter_delimiter)
fourier = scipy.fft.fft(signal)

plt.title("Signal noise")
plt.plot(time_arr, signal)
plt.show()

freq_arr = scipy.fft.fftfreq(len(time_arr))

plt.title("DFT_slow noise")
plt.plot(freq_arr, np.abs(X))
plt.show()

plt.title("fft noise")
plt.plot(freq_arr, np.abs(fourier))
plt.show()

# 3 Task


def FFT(signal):
    n = len(signal)
    if n == 1:
        return signal
    even = FFT(signal[::2])
    odd = FFT(signal[1::2])

    factor = np.exp(-2j * np.pi * np.arange(n//2) / n)
    result = np.zeros(n, dtype = complex)
    for k in range(n//2):
        result[k] = even[k] + factor[k] * odd[k]
        result[k + n//2] = even[k] - factor[k] * odd[k]
    return result

fast_fourier = FFT(signal_cos)
freq_arr = scipy.fft.fftfreq(len(fast_fourier), freq_1)

fourier = scipy.fft.fft(signal_cos)

plt.title("My FFT")
plt.plot(freq_arr, np.abs(fast_fourier))
plt.show()

