import numpy as np
import pywt
import matplotlib.pyplot as plt
import sys

# Setting
sampling_freq = 50  # [Hz]
sampling_interval = 1.0 / sampling_freq  # [sec]
data_num = 2500
base_frequencies = [0.2, 0.4, 0.8, 1.6, 3.2]

wavelet_name = 'gaus1'
cmap = 'cividis'

# Preparing input data
time_array = np.linspace(0.0, sampling_interval * data_num, data_num)
input_data = np.sin(2.0 * np.pi * time_array)
num_input_freqs = len(base_frequencies)
for i, freqs in enumerate(base_frequencies):
    input_data[int(i * data_num / num_input_freqs):int((i + 1) * data_num / num_input_freqs)] = \
        np.sin(2.0 * freqs * np.pi *
               time_array[int(i * data_num / num_input_freqs):int((i + 1) * data_num / num_input_freqs)])


def plot_input_data(axis, x, y):
    axis.set_title('input data')
    axis.plot(x, y)
    axis.set_xlabel('Time [sec]')
    axis.set_ylabel('Signal')
    axis.set_xlim(np.min(x), np.max(x))


def plot_wavelet_data(axis, time, freq, coef, title, xscale, color=cmap):
    xx, yy = np.meshgrid(time, freq)

    axis.set_title(title)
    axis.pcolormesh(xx, yy, coef, shading='nearest', cmap=color)
    axis.set_xlabel('Time [sec]')
    axis.set_ylabel('Frequencies [Hz]')
    axis.set_yscale(xscale)


def plot_results(time, input_sig, freq, coef, title='Wavelet results', xscale='linear'):
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plot_input_data(ax1, time, input_sig)
    plot_wavelet_data(ax2, time, freq, coef, title, xscale)
    plt.tight_layout()
    plt.show()
    plt.close()


def simple_wavelet_transform(input_sig, wavelet=wavelet_name):
    scale_list_linear = np.linspace(1.0, 1000, 100)
    coef, freq = pywt.cwt(input_sig, scale_list_linear, wavelet=wavelet, sampling_period=sampling_interval)
    plot_results(time_array, input_sig, freq, coef, title=sys._getframe().f_code.co_name)


simple_wavelet_transform(input_data)


def wavelet_transform_logplot(input_sig, wavelet=wavelet_name):
    scale_list_linear = np.linspace(1.0, 1000, 100)
    coef, freq = pywt.cwt(input_sig, scale_list_linear, wavelet=wavelet, sampling_period=sampling_interval)
    plot_results(time_array, input_sig, freq, coef, title=sys._getframe().f_code.co_name, xscale='log')


wavelet_transform_logplot(input_data)

