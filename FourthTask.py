import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft

# 1 Task

# Quadro
T = 2
amplitude = 1
N = 2
N_rect = 1000
sample_rate_rect = 100


def FuncRect(x):
    if x % T < 0.5 * T:
        return amplitude
    return -amplitude


def InitRectFunc(f):
    signal_ = list()
    time_arr = list()
    for i in range(N_rect):
        signal_.append(f(i / sample_rate_rect))
        time_arr.append(i / sample_rate_rect)
    return signal_, time_arr


A = 1

rec_signal, time = InitRectFunc(FuncRect)

kernel1 = list(map(lambda x: np.exp(-x ** 2), time))
kernel2 = list(map(lambda x: 2 * x if x < 1 else 0, time))

fig, axs = plt.subplots(5)
axs[0].set_title("Numpy Convolve")
axs[0].plot(time, rec_signal)
axs[1].plot(time, kernel1)
axs[2].plot(time, np.convolve(rec_signal, kernel1)[:N_rect])
axs[3].plot(time, kernel2)
axs[4].plot(time, np.convolve(rec_signal, kernel2)[:N_rect])
plt.show()


# 2 Task


def MyConvolve(signal, kernel):
    len1 = len(signal)
    len2 = len(kernel)
    length = len1 + len2 - 1
    y = np.zeros(length)

    for i in range(length):
        for j in range(len1):
            if 0 <= i - j < len2:
                y[i] += signal[j] * kernel[i - j]
    return y


fig, axs = plt.subplots(5)
axs[0].set_title("My Convolve")
axs[0].plot(time, rec_signal)
axs[1].plot(time, kernel1)
axs[2].plot(time, MyConvolve(rec_signal, kernel1)[:N_rect])
axs[3].plot(time, kernel2)
axs[4].plot(time, np.convolve(rec_signal, kernel1)[:N_rect])
plt.show()

# 3 Task

fig, axs = plt.subplots(3)
axs[0].set_title("Theorem of convolve")
axs[0].plot(time, rec_signal)
axs[1].plot(time, np.convolve(rec_signal, kernel1)[:N_rect])
axs[1].plot(time[::10], np.real(ifft(fft(rec_signal) * fft(kernel1)))[::10], 'o')
axs[2].plot(time, np.convolve(rec_signal, kernel2)[:N_rect])
axs[2].plot(time[::10], np.real(ifft(fft(rec_signal) * fft(kernel2)))[::10], 'o')
plt.show()

# 4 Task


def GaussianWindow(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    window = np.exp(-x**2 / (2 * sigma**2))
    return window / np.sum(window)


# create signal
sample_rate = 1000
n = 10000
p = 15
# poles for random interpolation
# noise level, measured in standard deviations
noiseamp = 5

# amplitude modulator and noise level
ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p)*30)
noise = noiseamp * np.random.randn(n)
signal1 = ampl + noise
# subtract mean to eliminate DC
signal1 = signal1 - np.mean(signal1)

gaus_window = GaussianWindow(n // 100, 5)
res_conv_time = np.convolve(signal1, gaus_window)

plt.plot(signal1, label='Signal')
plt.plot(res_conv_time, label='Convolution Result with Gaus')
plt.xlabel("time")
plt.legend()
plt.show()

# 5 Task
# Part a


def FreqGaus(p, fwhm, hz):
    s = fwhm * (2 * np.pi - 1) / (4 * np.pi)
    g = np.exp(-.5 * ((hz - p) / s) ** 2)
    return g


hz = np.linspace(0, sample_rate, n)

g = FreqGaus(100, 50, hz)
filteredSignal1 = np.real(ifft(g * fft(signal1)))

plt.plot(hz, np.abs(fft(signal1)), 'rs-', label='Signal')
plt.plot(hz, np.abs(fft(filteredSignal1)), 'bo-', label='Conv. result')
plt.ylim(0, 1000)
plt.xlim(0, 200)
plt.legend()
plt.show()

plt.plot(signal1, label='Signal')
plt.plot(filteredSignal1, label='Narrowband')
plt.xlabel("frequency")
plt.legend()
plt.show()

# Part b

g = FreqGaus(0, 50, hz)
filteredSignal2 = np.real(ifft(g * fft(signal1)))

plt.plot(signal1, label='Signal')
plt.plot(filteredSignal2, label='LowPass Filter')
plt.xlabel("frequency")
plt.legend()
plt.show()

# 6 Task


def PlanckTaper(t, N, e):
    planckTaper = np.zeros(len(t))
    for i in range(len(t)):
        if 0 < t[i] < (N - 1):
            zl = e * (N - 1) * (1/t[i] + 1/(t[i] - e * (N - 1)))
            planckTaper[i] = 1 / (np.exp(zl) + 1)
        if e * (N - 1) <= t[i] <= (1 - e) * (N - 1):
            planckTaper[i] = 1
        if (1 - e) * (N - 1) < t[i] < (N - 1):
            zr = e * (N - 1) * (1 / (N - 1 - t[i]) + 1 / ((1 - e) * (N - 1) - t[i]))
            planckTaper[i] = 1 / (np.exp(zr) + 1)
    return planckTaper


plt.plot(hz[:100], FreqGaus(0, 10, hz)[:100])
plt.plot(hz[:100], PlanckTaper(hz, 10, 0.01)[:100])
plt.ylabel('Gain')
plt.xlabel('Frequancy')
plt.show()

g = FreqGaus(1, 10, hz)
pl = PlanckTaper(hz, 10, 0.1)

filteredSignal1 = np.real(np.fft.ifft(g * fft(signal1)))
filteredSignal2 = np.real(np.fft.ifft(pl * fft(signal1)))

plt.plot(signal1)
plt.plot(filteredSignal1, label='Gaus')
plt.plot(filteredSignal2, label='Plank')
plt.legend()
plt.show()
