import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from mpl_toolkits.axes_grid1 import ImageGrid
import time
import os
import matplotlib.dates as mdates
from matplotlib.colorbar import Colorbar

file_bedroom = 'bedroom.csv'
file_living = 'living.csv'

png_dir = "png"
if not os.path.isdir(png_dir):
    os.mkdir(png_dir)


def wavelet_analysis(file, wavename, cmap):
    room_name = file.split(".")[0]
    dt = 1.0/48.0

    df = pd.read_csv(file)
    df['Date time'] = pd.to_datetime(df['Date time'])
    n_data = len(df)
    t = np.linspace(0.0, dt*n_data, n_data)

    scale_list = np.power(2, np.arange(-1, 18, 0.1))

    wavelet_time = time.time()

    coef_t, freqs_t = pywt.cwt(df['Temperature'], scale_list, wavelet=wavename, method='fft', sampling_period=dt)
    coef_p, freqs_p = pywt.cwt(df['Pressure']   , scale_list, wavelet=wavename, method='fft', sampling_period=dt)
    coef_h, freqs_h = pywt.cwt(df['Humidity']   , scale_list, wavelet=wavename, method='fft', sampling_period=dt)

    wavelet_time = time.time() - wavelet_time
    print("{} sec for wavelet transform".format(wavelet_time))

    fig = plt.figure(figsize=(12, 8), dpi=300)

    grid1 = ImageGrid(fig, 111,
                      nrows_ncols=(3, 1),
                      share_all=True,
                      aspect=False,
                      cbar_mode='single',
                      cbar_location='right',
                      cbar_pad=0.1,
                      cbar_size='0.5%'
                      )

    '''
    grid0 = ImageGrid(fig, (0.05, 0.05, 0.35, 0.9),
                      nrows_ncols=(3, 1),
                      share_all=False,
                      aspect=False
                      )

    grid0[0].plot(df['Date time'], df['Temperature'], color='k')
    grid0[1].plot(df['Date time'], df['Pressure'], color='k')
    grid0[2].plot(df['Date time'], df['Humidity'], color='k')
    daysFmt = mdates.DateFormatter('%m/%d')
    grid0[2].xaxis.set_major_formatter(daysFmt)
    fig.autofmt_xdate()
    '''

    rescale_coef_t = coef_t
    rescale_coef_p = coef_p
    rescale_coef_h = coef_h
    for i, scale in enumerate(scale_list):
        rescale_coef_t[i, :] = abs(coef_t[i, :]) / np.sqrt(scale)
        rescale_coef_p[i, :] = abs(coef_p[i, :]) / np.sqrt(scale)
        rescale_coef_h[i, :] = abs(coef_h[i, :]) / np.sqrt(scale)

    xx, yy = np.meshgrid(t, freqs_t)
    grid1[0].pcolormesh(xx, yy, rescale_coef_t, shading='nearest', cmap=cmap)
    grid1[0].set_yscale('log')

    grid1[1].pcolormesh(xx, yy, rescale_coef_p, shading='nearest', cmap=cmap)
    grid1[1].set_yscale('log')

    img = grid1[2].pcolormesh(xx, yy, rescale_coef_h, shading='nearest', cmap=cmap)
    grid1[2].set_yscale('log')

    grid1.cbar_axes[0].colorbar(img)

    # plt.show()
    plt.savefig(png_dir + "/{0}_with_wavelet_{1}_cmap_{2}.png".format(room_name, wavename, cmap), dpi=400)
    plt.close()


elapsed_time = time.time()

wavelet_list = []
for i in range(1, 9):
    wavelet_list.append("gaus{0}".format(i))
cmap_list = ['cividis'] # 'viridis',
file_list = [file_living, file_bedroom]
for file in file_list:
    for w in wavelet_list:
        for c in cmap_list:
            print("{0} : {1} : {2} running".format(file, w, c))
            wavelet_analysis(file, w, c)

elapsed_time = time.time() - elapsed_time
print("{0} sec passed".format(elapsed_time))


