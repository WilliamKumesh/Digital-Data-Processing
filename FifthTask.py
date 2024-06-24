import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
# 1 Task


def MorletWavelet(_time, _alpha):
    return np.exp(-1 * _time ** 2 / _alpha ** 2) * np.exp(2j * np.pi * _time)


def FourierMorletWavelet(_w, _alpha):
    return _alpha * np.sqrt(np.pi) * np.exp(((-_alpha**2)*(2*np.pi-_w)**2)/4)


N = 1000
time = np.linspace(-6, 6, N)
alpha = 2
morlet = MorletWavelet(time, alpha)

plt.title("MorletWavelet")
plt.plot(time, morlet)
plt.show()

w = np.linspace(0, 10, N)
morletFreq = FourierMorletWavelet(w, alpha)

plt.plot(w, morletFreq)
plt.show()

# 2 Task


def MexicanHatWavelet(_time):
    return (1 - _time**2) * np.exp(-1 * _time**2 / 2)


mhw = MexicanHatWavelet(time)

plt.title("MexicanHat")
plt.plot(time, mhw)
plt.show()

plt.plot(np.fft.fftfreq(len(time), d=1/100)[:100], np.abs(fft(mhw))[:100])
plt.xlim(0, 2)
plt.show()

# 3 Task


def HaarWavelet(_time):
    return np.array([1 if 0.5 > i >= 0 else -1 if 1 > i >= 0.5 else 0 for i in _time])


haar = HaarWavelet(time)

plt.title("HaarWavelet")
plt.plot(time, haar)
plt.show()

plt.plot(np.fft.fftfreq(len(time), d=1/100)[:100], np.abs(fft(haar))[:100])
plt.show()

# Task 4

#ampl_noise = 0.1
noize = np.random.random(N)
N = 1000
fs = 10000
t = np.arange(N) / fs

cos_50 = np.cos(2 * np.pi * 50 * t)
cos_150 = np.cos(2 * np.pi * 150 * t)

cos = cos_50 + cos_150 + noize - np.mean(noize)

filteredSignal1 = np.convolve(cos, morlet)[N//2: -N//2]
filteredSignal2 = np.convolve(cos, mhw)[N//2: -N//2]
filteredSignal3 = np.convolve(cos, haar)[N//2: -N//2]

freq = np.fft.fftfreq(len(cos), 1/fs)

plt.plot(t, cos)
plt.show()
fig, axs = plt.subplots(4)
axs[0].plot(freq[:N//2], np.abs(fft(cos))[:N//2], color='r', label='Fourier')
axs[0].legend()
axs[0].set_xlim(0, 500)
axs[1].plot(freq[:N//2], np.abs(fft(filteredSignal1))[:N//2], color='r', label='Morlet')
axs[1].legend()
axs[1].set_xlim(0, 500)
axs[2].plot(freq[:N//2], np.abs(fft(filteredSignal2))[:N//2], color='g', label='MexicanHat')
axs[2].legend()
axs[2].set_xlim(0, 500)
axs[3].plot(freq[:N//2], np.abs(fft(filteredSignal3))[:N//2], color='b', label='Haar')
axs[3].legend()
axs[3].set_xlim(0, 500)
plt.show()

# 5 Task

sampleRate = 1000
time = np.arange(-3, 3, 1/sampleRate)
N = len(time)

freqmod = np.exp(-time ** 2)*10+10
freqmod = freqmod + np.linspace(0, 10, N)
signal = np.sin(2 * np.pi * (time + np.cumsum(freqmod)/sampleRate))

# plt.plot(time, signal)
# plt.show()

nfreq = 50
freq = np.linspace(3, 35, nfreq)
fwhm = 0.2

wavelets = np.zeros((nfreq, N), dtype=complex)

for w in range(nfreq):
    g = np.exp(-(4*np.log(2) * time**2)/fwhm**2)
    wavelets[w] = np.exp(2j*np.pi*freq[w] * time) * g

nconv = N*2 - 1
halfk = N//2 + 1

sigX = fft(signal, nconv)
tf = np.zeros((nfreq, N), dtype=float)

wavelet = MorletWavelet(time, 0.25)
_waveX = fft(wavelet, nconv)
_waveX = _waveX/np.max(_waveX)
_converse = ifft(_waveX * sigX)
_converse = np.real(_converse)[N//2: - N//2 + 1]**2

for f in range(nfreq):
    waveX = fft(wavelets[f], nconv)
    waveX = waveX/np.max(waveX)
    convres = ifft(waveX * sigX)
    convres = convres[N//2: - N//2 + 1]
    tf[f] = np.abs(convres)**2

plt.contourf(time, freq, tf)
plt.show()

# 6 Task


# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy.io.wavfile as wavfile

Fs, aud = wavfile.read('Lya.wav')
# select left channel only
aud = aud[:,0]
# trim the first 125 seconds
N = len(aud)
time = np.linspace(0, N, 1000)

morlet = MorletWavelet(time, alpha)
mhw = MexicanHatWavelet(time)
haar = HaarWavelet(time)

filteredSignal1 = np.convolve(aud, morlet)[:N//2]
filteredSignal2 = np.convolve(aud, mhw)[:N//2]
filteredSignal3 = np.convolve(aud, haar)[:N//2]

result1 = np.abs(fft(filteredSignal1))
result2 = np.abs(fft(filteredSignal2))
result3 = np.abs(fft(filteredSignal3))

freq = np.fft.fftfreq(N//2, 1/Fs)

fig, axs = plt.subplots(3)
axs[0].plot(freq, result1, color='r', label='Morlet')
axs[0].legend()
axs[0].set_xlim(0, 500)
axs[1].plot(freq, result2, color='g', label='MexicanHat')
axs[1].legend()
axs[1].set_xlim(0, 500)
axs[2].plot(freq, result3, color='b', label='Haar')
axs[2].legend()
axs[2].set_xlim(0, 500)
plt.show()

plt.figure(figsize=(15,8))
plt.specgram(filteredSignal1, Fs=Fs, cmap='terrain')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0, 2000])
plt.show()

plt.figure(figsize=(15,8))
plt.specgram(filteredSignal2, Fs=Fs, cmap='brg')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0, 2000])
plt.show()

plt.figure(figsize=(15,8))
plt.specgram(filteredSignal3, Fs=Fs, cmap='turbo_r')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0, 2000])
plt.show()