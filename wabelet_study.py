import numpy as np
import pywt
import matplotlib.pyplot as plt

# physcal parameter
sampling_freq = 50  # Hz
sampling_interval = 1.0 / sampling_freq  # second

# preparing sample data
data_num = 2000
time_array = np.linspace(0.0, sampling_interval*data_num, data_num)
input_data = np.sin(np.pi * time_array)
input_data[int(0.25*data_num):int(0.50*data_num)] = np.sin(2.0 * np.pi * time_array[int(0.25*data_num):int(0.50*data_num)])
input_data[int(0.50*data_num):int(0.75*data_num)] = np.sin(4.0 * np.pi * time_array[int(0.50*data_num):int(0.75*data_num)])
input_data[int(0.75*data_num):int(1.00*data_num)] = np.sin(8.0 * np.pi * time_array[int(0.75*data_num):int(1.00*data_num)])

scale_list_linear = np.arange(1, 1000)
coef, freqs = pywt.cwt(input_data, scale_list_linear, wavelet='gaus1', method='fft', sampling_period=sampling_interval)

# showing results
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_title('input data')
ax1.plot(time_array, input_data)
ax1.set_xlabel('time [sec]')
ax1.set_ylabel('signal')

ax2 = fig.add_subplot(212)
xx, yy = np.meshgrid(time_array, freqs)
ax2.set_title("simple transform")
ax2.pcolormesh(xx, yy, coef, shading='nearest', cmap='cividis')
ax2.set_xlabel('time [sec]')
ax2.set_ylabel('freqs')

plt.tight_layout()
plt.show()
plt.close()
