'''
Useful function for Welch percentile estimation
'''

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib

def compute_nu(N, Ns, win, o):
    """
    Computes the equivalent degree of freedom for a given sequence lentgh,
    segment length, window, and overlap. The equation is taken from p. 429 of
    Percival & Walden "Spectral analysis for univariate time series" (2020)

    N : int
        number of samples in signal
    Ns : int
        number of samples in a segment
    win : array
        window function
    o : percentage of overlap between adjacent segments

    returns : float
        equivalent degree of freedom
    """

    Nb = int(N/(Ns*(1-o))) # number of segments
    
    sum_denom = 0
    for m in np.linspace(1, Nb-1, Nb-1, dtype=int):
        sum_denom += (1 - m/Nb)*_sum_win(win, Ns, m, o)
    nu = 2*Nb / (1 + 2 * sum_denom)
    return nu

def _sum_win(win, Ns, m, o):
    """
    Helper function for computing the equivalent degree of freedom
    """
    win_s = np.roll(win, -int((1-o)*Ns)*m)
    win_s[-int((1-o)*Ns)*m:] = 0
    return abs(np.sum(win * win_s))**2

# parts are from https://github.com/scipy/scipy/blob/v1.5.2/scipy/signal/spectral.py#L291-L454
def welch_percentile(x, bias_func=bias_digamma_approx, fs=1.0, window='hann',
                     nperseg=None, overlap=None, nfft=None, detrend='constant',
                     return_onesided=True, scaling='density', axis=-1,
                     percentile=0.5, numRV='edof', percentage_outliers=0.0):
    """
    This function implements the Welch Percentile (WP) estimator. It extends
    the scipy.signal.welch() function by supporting arbitrary percentiles
    (instead of just median) and also uses a more accurate bias correction
    term. The mathematical background along with extensive simulations can be 
    found in Schwock & Abadi "Statistical properties of a modified Welch method
    that uses sample percentiles" (2021)

    x : array
        signal in time domain
    bias_func : approximation used for the bias correction. All functions are
        implemented below. Default is digamma approximation as this works well
        for arbitrary percentiles and number of segments. Other functions can
        give faster performance, however at the cost of lower accuracy.
    overlap : float
        percentage of overlap between adjacent segments
    percentile : float between 0 to 1
        percentile (as fraction between 0 to 1) used for the estimation. If
        percentile is None, mean averaging over periodograms is used.
    numRV : str
        number of independent random variables assumed for the percentile
        estimation. Since adjacent segments are allowed to overlap, periodogram
        estimates are not necessarily independent. If 'edof' (default) is used
        this is accounted for by computing the equivalent number of independent
        periodgoram which improves the bias correction. If independence between
        periodograms can be guaranteed 'n', the number of segments, can also be
        used.
    percentage_outliers : float between 0 to 1
        approximate percentage of outliers in the data. This parameter is
        useful if the percentage of outlier is large (>5%) and known and an
        unbiased estimate is desired.

    All other parameters are the same as for the scipy.signal.welch() function
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html

    retuns : (freqs, Pxy, Nb)
        freqs : array
            frequency bins
        Pxy : array
            power spectral density estimate
        Nb : int
            number of segments. This parameter can usually be ignored but may
            be useful for debugging and analysis purpose.

    """

    freqs, _, Pxy = signal.spectral._spectral_helper(x, x, fs, window, nperseg, int(nperseg * overlap), nfft,
                                            detrend, return_onesided, scaling, axis,
                                            mode='psd')
    
    Nb = Pxy.shape[-1]

    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            if percentile == None:
                Pxy = Pxy.mean(axis=axis)
            elif isinstance(percentile, int) or isinstance(percentile, float):
                if percentile >= 0 and percentile <= 1:
                    
                    if numRV == 'n':
                        Pxy = np.quantile(Pxy, percentile, axis=axis) / bias_func(Pxy.shape[-1], percentile,
                                                                                percentage_outliers)
                    elif numRV == 'edof':
                        win = signal.get_window(window, nperseg) / np.sqrt(np.sum(signal.get_window(window, nperseg)**2))
                        edof = wp.compute_nu(len(x), nperseg, win, overlap)
                        Pxy = np.quantile(Pxy, percentile, axis=axis) / bias_func(edof / 2, percentile,
                                                                                    percentage_outliers)
                else:
                    raise ValueError('percentile must be between 0 and 1, got %s'
                                     % (percentile,))
            else:
                raise ValueError('percentile must be integer, float, or None type, got type %s'
                                     % (type(percentile),))
        else:
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])
    
    return freqs, Pxy, Nb

# below are the differnt functions for estimating the bias of the quantile
# estimate. all bias functios take Nb (the number of segments or the number of
# equivalent independent segments (i.e., EDOF/2)), p (the percentile between 0
# to 1), and percentage_outliers (the percentage of outliers in the data) as
# inputs. Some bias formulas don't use all input parameters 

def bias_alternating_harmonic_series(Nb, p, percentage_outliers):
    """
    Bias according to Allen at al. "Findchirp: An algorithm for detection of
    gravitational waves from inspiraling compact binaries” (2012). The bias is
    only accurate for large Nb or odd numbers of Nb
    """
    Nb = np.round(Nb) # Nb has to be an integer
    l = np.linspace(1, Nb, Nb)
    return np.sum((-1)**(l+1)/l)

def bias_truncated_harmonic_series(Nb, p, percentage_outliers):
    """
    This formula is an improvement over the alternating harmonic series.
    """
    Nb = Nb * (1 - percentage_outliers)
    Nb = np.round(Nb) # Nb has to be an integer
    p = p / (1 - percentage_outliers)

    percentiles = np.round(np.linspace(1/(Nb+1), 1 - 1/(Nb+1), Nb), 3)
    if p in percentiles:
        l = np.arange(np.round((Nb-1)*(1-p)) + 1, Nb+1, 1)
    else:
        l = np.arange(np.round(Nb*(1-p)) + 1, Nb+2, 1)
    return np.sum(1/l)

def bias_digamma_approx(Nb, p, percentage_outliers):
    """
    This is the most general formula for the bias which is accurate for all
    percentiles and also small number of segments Nb
    """
    Nb = Nb * (1 - percentage_outliers)
    p = p / (1 - percentage_outliers)

    return digamma(Nb+2) - digamma(Nb*(1-p)+1)

def digamma(x):
    """
    approximation of digamma function
    """
    return np.log(x) - 1/(2*x) - 1/(12*x**2) + 1/(120*x**4) - 1/(252*x**6)

def bias_limit(Nb, p, percentage_outliers):
    """
    Bias in the limit, that is for Nb -> inf. This is arguably the fastest
    method but is only accurate for large enough Nb.
    """
    p = p / (1 - percentage_outliers)

    return -np.log(1-p)

def no_bias_correct(Nb, p, percentage_outliers):
    """
    No bias correction. This is useful for evaluating the performance of the
    other formulas
    """
    return 1.0


def var_welch_percentile(Nb, p, bias, percentage_outliers):
    """
    This estimates the variance of the percentile estimator assuming that the
    true power spectral density is 1 (e.g., in the case of white Gaussian noise
    with variance 1) and using the trigamma approximation formula.
    
    The function takes the bias as an additional input  parameter 
    """
    Nb = Nb * (1 - percentage_outliers)
    p = p / (1 - percentage_outliers)

    var_beta = trigamma(Nb*(1-p) + 1) - trigamma(Nb+2)
    var = (1/bias)**2 * var_beta
    
    return var

def trigamma(x):
    """
    approximation of trogamma function
    """
    return 1/x + 1/(2*x**2) + 1/(6*x**3) - 1/(30*x**5) + 1/(42*x**7)

def var_welch_percentile_limit(Nb, p, bias, percentage_outliers):
    """
    This estimates the variance of the percentile estimator assuming that the
    true power spectral density is 1 (e.g., in the case of white Gaussian noise
    with variance 1) and using the limit approximation.
    
    The function takes the bias as an additional input  parameter 
    """
    Nb = Nb * (1 - percentage_outliers)
    p = p / (1 - percentage_outliers)

    var = 1/(bias)**2  * p / (Nb * (1-p))
    return var