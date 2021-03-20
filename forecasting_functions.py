#################################################################################################################################################
#                                                                                                                                               #
#                                                        IMPORTING PACKAGES                                                                     #
#                                                                                                                                               #
#################################################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from scipy import signal
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf


#################################################################################################################################################
#                                                                                                                                               #
#                                                        FIRST FUNCTION LAYER                                                                   #
#                                                                                                                                               #
#################################################################################################################################################
def sortBy(l1, l2):
    """Sorts l1 with the l2 values"""
    n = len(l1)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if l2[j] < l2[j + 1]:
                l2[j], l2[j + 1] = l2[j + 1], l2[j]
                l1[j], l1[j + 1] = l1[j + 1], l1[j]
    return l1

def getSeasonnalPeriod(serie, order=1, filter_acf=None, nlags=None, show=False):
    """return period to use for seasonnal decomposition. Uses the autocorrelation function"""

    if nlags == None:
        nlags = len(serie)

    # Get autocorrelation function of signal
    f = acf(serie, nlags=nlags, fft=True)

    if filter_acf != None:
        # Low-pass filter f
        b, a = signal.butter(3, filter_acf, btype='lowpass')
        f = signal.filtfilt(b, a, f)

    indexs = [i for i in range(len(f))]

    # Get local maximas indexs
    maxi = list(argrelextrema(f, np.greater_equal, order=order)[0])

    # Remove start and end : Not really local maximas
    if 0 in maxi:
        maxi.remove(0)
    if indexs[-1] in maxi:
        maxi.remove(indexs[-1])

    # Sort by amplitudes
    amplitudes = [f[i] for i in maxi]
    periodsToAmpls = {maxi[i]: amplitudes[i] for i in range(len(maxi))}

    sorted_periods = sortBy(list(periodsToAmpls.keys()), amplitudes)

    if show:
        plt.plot(indexs, f)
        plt.show()

    if len(sorted_periods) == 0:
        return 0

    return sorted_periods


#################################################################################################################################################
#                                                                                                                                               #
#                                                       SECOND FUNCTION LAYER                                                                   #
#                                                                                                                                               #
#################################################################################################################################################    
def extractTrend(signal, model_trend=Ridge(), trend_degree=3, moving_window=None, moving_window_degree=1, N=1):
    """Return trend of signal"""
    y = signal
    x = [[i] for i in range(len(list(y)))]

    # Interpolate signal with model_trend to get the trend
    model = make_pipeline(PolynomialFeatures(degree=trend_degree), StandardScaler(), model_trend)
    model.fit(x, y)
    trend = model.predict(x)

    # Forecast trend
    if moving_window == None:
        # Use same model
        pred_trend = model.predict([[i + len(list(y))] for i in range(N)])
    else:
        # Fit model on last moving_window values of signal
        y = signal[-moving_window:]
        x = [[i] for i in range(len(list(y)))][-moving_window:]
        model = make_pipeline(PolynomialFeatures(degree=trend_degree), StandardScaler(), model_trend)
        model.fit(x, y)
        pred_trend = model.predict([[i + len(list(y))] for i in range(N)])
    
    return trend, signal-trend, pred_trend


    
def extractSeasonnalities(serie, model_seasonnal=RandomForestRegressor(), n=None, Kbests=None, treshold=0, N=1, verbose=False):
    """Get different seasonnalities of signal"""
    y = serie

    # Create n = periodic components (features for models)
    if n==None:
        # Classical periods
        n = [2, 7, 12, 24, 60, 30, 365]
        # Automatically find good periods (local maximas of autocorrelation function)
        p = getSeasonnalPeriod(serie, order=1, filter_acf=None, show=False)[:3]
        p = [i for i in p if i!=0]
        n += p
        # Avoid overfitting by removing largest periods
        n = [i for i in n if 2*i<=len(y)]
    n = list(dict.fromkeys(n))
    # Sort by ascending order is important
    n.sort()


    # Extract seasonnalities for each n    
    seasonnalities = {}
    pred_seasonnalities = {}
    y_dt = list(y)
    for p in n:
        x = [[i%p] for i in range(len(list(y)))]
        model = make_pipeline(StandardScaler(), model_seasonnal)
        model.fit(x, y)
        seasonnalities[str(p)] = model.predict(x)
        pred_seasonnalities[str(p)] = model.predict([[(i+len(list(y)))%p] for i in range(N)])

        # Remove p-seasonnality from signal
        y = y-seasonnalities[str(p)]
        

    # Get score for each se asonnality, keep only if greater than treshold
    periods = []
    variances = []
    if verbose:
        print('  {: <10}'.format('Period '),'Variance')
    for p in seasonnalities:
        # Filter y to compare variances of same frequencies
        b, a = signal.butter(3, 1/int(p), btype='highpass')
        y_t = signal.filtfilt(b, a, list(serie))
        y_dt = serie - y_t

        # Compute y variance
        mean_y = sum(y_dt) / len(y_dt) 
        variance_y = sum([((x - mean_y) ** 2) for x in y_dt]) / len(y_dt)

        # Compute p-seasonnality relative variance
        mean = sum(seasonnalities[p]) / len(seasonnalities[p]) 
        variance = (sum([((x - mean) ** 2) for x in seasonnalities[p]]) / len(seasonnalities[p]))/variance_y

        if verbose:
            print('  {:.<10}'.format(p+' '),round(variance,3))
    
        if variance > treshold:
            periods.append(int(p))
            variances.append(variance)

    # Keep only K bests seasonnalities
    if Kbests != None:
        periods = sortBy(periods, variances)
        periods = periods[:Kbests]

    seasonnalities = {str(p) : seasonnalities[str(p)] for p in periods[:Kbests]}
    pred_seasonnalities = {str(p) : pred_seasonnalities[str(p)] for p in periods[:Kbests]}

    return seasonnalities, pred_seasonnalities




        


