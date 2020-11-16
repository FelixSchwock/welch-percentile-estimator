'''
Useful function for Welch percentile estimation
'''

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib

def compute_nu(N, Ns, win, o):
    Nb = int(N/(Ns*(1-o)))
    def _sum_win(win, Ns, m, o):
        win_s = np.roll(win, -int((1-o)*Ns)*m)
        win_s[-int((1-o)*Ns)*m:] = 0
        return abs(np.sum(win * win_s))**2
    sum_denom = 0
    for m in np.linspace(1, Nb-1, Nb-1, dtype=int):
        sum_denom += (1 - m/Nb)*_sum_win(win, Ns, m, o)
    nu = 2*Nb / (1 + 2 * sum_denom)
    return nu

# parts are from https://github.com/scipy/scipy/blob/v1.5.2/scipy/signal/spectral.py#L291-L454
def welch_percentile(x, bias_func, fs=1.0, window='hann', nperseg=None, overlap=None, nfft=None,
          detrend='constant', return_onesided=True, scaling='density',
          axis=-1, percentile=None, numRV='edof'):

    freqs, _, Pxy = signal.spectral._spectral_helper(x, x, fs, window, nperseg, int(nperseg * overlap), nfft,
                                            detrend, return_onesided, scaling, axis,
                                            mode='psd')
    
    Nb = Pxy.shape[-1]

    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            if percentile == None:
                Pxy = Pxy.mean(axis=-1)
            elif isinstance(percentile, int) or isinstance(percentile, float):
                if percentile >= 0 and percentile <= 1:
                    
                    if numRV == 'n':
                        Pxy = np.quantile(Pxy, percentile, axis=-1) / bias_func(Pxy.shape[-1], percentile)
                    elif numRV == 'edof':
                        win = signal.get_window(window, nperseg) / np.sqrt(np.sum(signal.get_window(window, nperseg)**2))
                        edof = compute_nu(len(x), nperseg, win, overlap)
                        if bias_func == bias_digamma_approx:
                            Pxy = np.quantile(Pxy, percentile, axis=-1) / bias_func(edof / 2, percentile)
                        else:
                            Pxy = np.quantile(Pxy, percentile, axis=-1) / bias_func(int(np.round(edof / 2)), percentile)
                else:
                    raise ValueError('percentile must be between 0 and 1, got %s'
                                     % (percentile,))
            else:
                raise ValueError('percentile must be integer, float, or None type, got type %s'
                                     % (type(percentile),))
        else:
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])
    
    return freqs, Pxy, Nb

def bias_alternating_harmonic_series(N, p):
    l = np.linspace(1, N, N)
    return np.sum((-1)**(l+1)/l)

def bias_truncated_harmonic_series(N, p):
    percentiles = np.round(np.linspace(1/(N+1), 1 - 1/(N+1), N), 3)
    if p in percentiles:
        l = np.arange(np.round((N-1)*(1-p)) + 1, N+1, 1)
    else:
        l = np.arange(np.round(N*(1-p)) + 1, N+2, 1)
    return np.sum(1/l)

def bias_digamma_approx(N, p):
    def digamma(x):
        return np.log(x) - 1/(2*x) - 1/(12*x**2) + 1/(120*x**4) - 1/(252*x**6)
    
    return digamma(N+2) - digamma(N*(1-p)+1)

def no_bias_correct(N, p):
    return 1.0

def bias_limit(N, p):
    return -np.log(1-p)


def var_welch_percentile_wn(n, p, bias):
    def trigamma(x):
        return 1/x + 1/(2*x**2) + 1/(6*x**3) - 1/(30*x**5) + 1/(42*x**7)
    
    var_beta = trigamma(n*(1-p) + 1) - trigamma(n+2)
    var = (1/bias)**2 * var_beta
    
    return var

def var_welch_percentile_limit_wn(nu, p, bias):
    #var = 1/(bias)**2  * p / ((np.floor(nu/2) + 2) * (1-p))
    var = 1/(bias)**2  * p / ((np.floor(nu/2)) * (1-p))
    return var

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