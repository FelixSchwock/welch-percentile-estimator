import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib

class PxxPercentile:
    def __init__(self, freqs, p, values):
        self.freqs = freqs
        self.p = p
        self.values = values
        
class PxxDensity:
    def __init__(self, freqs, bin_edges, values):
        self.freqs = freqs
        self.bin_edges = bin_edges
        self.values = values


def welch_percentile_full(x, fs=1.0, window='hann', nperseg=None, overlap=None, nfft=None,
          detrend='constant', return_onesided=True, scaling='density',
          axis=-1, percentiles=None):
    freqs, _, Pxy = signal.spectral._spectral_helper(x, x, fs, window, nperseg, int(nperseg * overlap), nfft,
                                            detrend, return_onesided, scaling, axis,
                                            mode='psd')
    
    if isinstance(percentiles, type(None)):
        percentiles = np.arange(0.05, 1.0, 0.05)
        
    pxx_percentiles = np.array([None] * len(percentiles))

    for i, p in enumerate(percentiles):
        pxx_quant = np.quantile(Pxy, p, axis=-1)
        pxx_percentiles[i] = pxx_quant
        
    return PxxPercentile(freqs, percentiles, pxx_percentiles)

def spectral_probability_density(x, fs=1.0, window='hann', nperseg=None, overlap=None, nfft=None,
          detrend='constant', return_onesided=True, scaling='density', axis=-1, nbins=20):
    freqs, _, Pxy = signal.spectral._spectral_helper(x, x, fs, window, nperseg, int(nperseg * overlap), nfft,
                                            detrend, return_onesided, scaling, axis,
                                            mode='psd')
    
    pxx_density = []
    hist_min = 10 * np.log10(np.quantile(Pxy, 0.001))
    hist_max = 10 * np.log10(np.quantile(Pxy, 1)) + 10
    bins = np.linspace(hist_min, hist_max, nbins)
    for row in Pxy:
        hist, _ = np.histogram(10 * np.log10(row), bins=bins, density=True)
        pxx_density.append(hist)

    pxx_density = np.array(pxx_density)
    return PxxDensity(freqs, bins, pxx_density)


def plot_percentile(pxx_percentile, pmarks=[0.1, 0.5, 0.9], colors=('blue', 'red', 'red')):
    Ns = len(pxx_percentile.freqs)
    freqs = pxx_percentile.freqs
    
    for i, pxx in enumerate(pxx_percentile.values):
        p = np.round(pxx_percentile.p[i], 2)
        if p in pmarks:
            plt.annotate(str(np.round(p*100, 2)) + '%', (freqs[int(0.4*Ns)],
                                                         10 * np.log10(pxx[int(0.4*Ns)])), color=colors[2])
            plt.plot(freqs[:int(Ns/2)], 10 * np.log10(pxx[:int(Ns/2)]), color=colors[1])
        else:
            plt.plot(freqs[:int(Ns/2)], 10 * np.log10(pxx[:int(Ns/2)]), color=colors[0])
        

    
    #plt.xlabel('freq')
    #plt.ylabel('power spectral density [dB]')
    #plt.title('Quanile power spectral density')
    plt.grid(True)

def plot_spectral_density(pxx_density, ax):
    Ns = len(pxx_density.freqs)
    freqs = pxx_density.freqs

    vdelta = 0.001
    vmin = 0
    vmax = np.max(pxx_density.values)

    cbarticks = np.arange(vmin, vmax+vdelta, vdelta)

    bins = (pxx_density.bin_edges[1:] + pxx_density.bin_edges[:-1]) / 2
    im = ax.contourf(freqs[:int(Ns/2)], bins, np.transpose(pxx_density.values[:int(Ns/2)]),
                    cbarticks, norm=colors.Normalize(vmin=0, vmax=0.2), cmap=plt.cm.jet)
    plt.colorbar(im, ax=ax, pad=0.1, label='probaility density')

    #plt.xlabel('freq')
    #plt.ylabel('power spectral density [dB]')
    #plt.title('Quanile power spectral density')
    plt.grid(True)