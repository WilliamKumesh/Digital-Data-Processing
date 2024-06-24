import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from matplotlib import pyplot


class FourierTransform:
    @staticmethod
    def calculate_coefficients(t0, T, signal_func, n):
        sin_func = lambda t: signal_func(t) * np.sin(t * n * 2 * np.pi / T)
        cos_func = lambda t: signal_func(t) * np.cos(t * n * 2 * np.pi / T)

        a_n = 2/T * integrate.quad(cos_func, t0, t0 + T)[0]
        b_n = 2/T * integrate.quad(sin_func, t0, t0 + T)[0]

        return a_n, b_n

    @staticmethod
    def calculate_a0(t0, T, signal_func):
        return 2/T * integrate.quad(signal_func, t0, t0 + T)[0]

    @staticmethod
    def calculate_fourier_series(t0, T, signal_func, N, t):
        a0 = FourierTransform.calculate_a0(t0, T, signal_func)
        result = 0
        result += a0/2
        for i in range(N):
            an, bn = FourierTransform.calculate_coefficients(t0, T, signal_func, i)
            result += an * np.cos(2 * np.pi * i * t / T) + bn * np.sin(2 * np.pi * i * t / T)
        return result


# Quadro
T = 2
amplitude = 1
N = 2
# Cos
freq = 100


def f(x):
    if x % T > 0.5 * T:
        return amplitude
    return -amplitude


def f_cos(x):
    return amplitude*np.cos(freq * x * 2 * np.pi)


def f_noise(x):
    if x % T > 0.5 * T:
        return amplitude + np.random.random(1)[0] / 2 * 15
    return -amplitude - np.random.random(1)[0] / 2 * 15


def f_cos_noise(x):
    return amplitude*np.cos(freq * x * 2 * np.pi) + np.random.random(1)[0] / 2 * 15


def print_time_quadro_graphics(t0, T, signal_func, N):
    fourier = list()
    signal = list()
    time = list()
    error = list()
    noise = list()
    for i in range(1000):
        iter = i / 100
        fourier.append(FourierTransform.calculate_fourier_series(t0, T, signal_func, N, iter))
        signal.append(signal_func(iter))
        noise.append(f_noise(iter))
        time.append(iter)
        error.append(signal[i] - fourier[i])

    fig, axs = plt.subplots(4)
    plt.subplots_adjust(wspace = 0, hspace = 2)
    axs[0].plot(time, signal)
    axs[0].set_title("Signal")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("amplitude")
    axs[1].plot(time, fourier)
    axs[1].set_title("Fourier")
    axs[2].plot(time, error)
    axs[2].set_title("Error")
    axs[3].plot(time, noise)
    axs[3].set_title("Noise")
    plt.show()

    plt.xlim(0, 0.06)
    plt.title("Spectrum")
    plt.magnitude_spectrum(fourier, 2/T)
    plt.show()
    plt.xlim(0, 0.06)
    plt.magnitude_spectrum(noise, 2 / T)
    plt.show()


def print_time_cos_graphics(t0, T, signal_func, N):
    fourier = list()
    signal = list()
    time = list()
    error = list()
    noise = list()
    for i in range(1000):
        iter = i / 10000
        fourier.append(FourierTransform.calculate_fourier_series(t0, T, signal_func, N, iter))
        signal.append(signal_func(iter))
        time.append(iter)
        error.append(signal[i] - fourier[i])
        noise.append(f_cos_noise(iter))

    fig, axs = plt.subplots(3)
    plt.subplots_adjust(wspace=0, hspace=2)
    axs[0].plot(time, signal)
    axs[0].set_title("Signal")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("amplitude")
    axs[0].set_xlabel("time")
    axs[1].plot(time, fourier)
    axs[1].set_title("Fourier")
    axs[2].plot(time, noise)
    axs[2].set_title("Noise")
    plt.show()

    plt.xlim(0, 3)
    plt.magnitude_spectrum(fourier, freq)
    plt.show()

    plt.xlim(0, 3)
    plt.magnitude_spectrum(noise, freq)
    plt.show()


def print_an_bn(t0, T, signal_func, N):
    an_list = list()
    bn_list = list()
    iter = list()

    for i in range(N):
        an, bn = FourierTransform.calculate_coefficients(t0, T, signal_func, i)
        an_list.append(an)
        bn_list.append(bn)
        iter.append(i)

    fig, axs = plt.subplots(2)
    #plt.subplots_adjust(wspace=0, hspace=2)
    axs[0].plot(iter, an_list)
    axs[0].set_title("An")
    axs[0].set_xlabel("iterations")
    axs[0].set_ylabel("amplitude")
    axs[1].plot(iter, bn_list)
    axs[1].set_title("Bn")
    plt.show()


print_time_quadro_graphics(0, T, f, N)
print_time_cos_graphics(0, 1/freq, f_cos, 4)
#print_an_bn(0, T, f, N)
#print_an_bn(0, T, f_cos, N)
