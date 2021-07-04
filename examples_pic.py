#################################################################################################################################################
#                                                                                                                                               #
#                                                        IMPORTING PACKAGES                                                                     #
#                                                                                                                                               #
#################################################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from forecasting_functions import *





#################################################################################################################################################
#                                                                                                                                               #
#                                                                   PARAMETERS                                                                  #
#                                                                                                                                               #
#################################################################################################################################################    
# Steps to forecast
N = 100

# Periods of signal. Can be find with autocorrelation function
# Can be determined with getSeasonnalPeriod()
periods = [6,12, 23]


# Normalize period : increse it gives mor importance to each values, it averages less the signel
# /!\ keep it greater than biggest period
# If > signal length, signal will be not normalized
NORMALIZE_TREND_PERIOD = 15
NORMALIZE_VAR_PERIOD = 15


# start_index : increase it gives more importance to last signel values
start_index_trend = 0
start_index_var = 0
start_index_seasonnal = 0

# Increase it to predict non-linear trends and variance variations.
trend_degree = 1
var_degree = 1

# Random state (No need to change)
random_state = 0





#################################################################################################################################################
#                                                                                                                                               #
#                                                               PREDICTION                                                                      #
#                                                                                                                                               #
#################################################################################################################################################    
SHOW = True


### Get data
dico = {}
data = {}
dicoFiles = {"data/Climate.csv": "Climate",
             "data/Electric_Production.csv": "Electric",
             "data/Beer.csv": "Beer",
             "data/Shampoo.csv": "Shampoo"}

for f in dicoFiles:
    dico[dicoFiles[f]] = pd.read_csv(f)
data['Temperatures'] = dico['Climate']['meantemp'].dropna()
data['Humidity'] = dico['Climate']['humidity'].dropna()
data['Electric'] = dico['Electric']['IPG2211A2N'].dropna()
data['Beer'] = dico['Beer']['Monthly beer production'].dropna()
data['Shampoo'] = dico['Shampoo']['Sales of shampoo over a three year period'].dropna()


FILE = "Beer.png"
TITLE = "Beer sales prediction"

df = pd.DataFrame()
df['y'] = pd.Series(data['Beer'])

df = df.reset_index()


df['y'].plot()
plt.savefig('init.png')
plt.show()




### NORMALIZE SIGNAL ###
df['trend'], df['var'], df['normalized'] = Normalize(df['y'],
                                                     rolling_period_trend=NORMALIZE_TREND_PERIOD,
                                                     rolling_period_var=NORMALIZE_VAR_PERIOD)

if SHOW:
    #df['normalized'].hist(bins=100)
    df['normalized'].plot()
    plt.title("Normalized signal")
    plt.savefig("Norm.png")
    plt.show()

a = [i for i in range(2, 400)]
p = findBestPeriods(df, a, col='normalized', strategy='median', show=True)
    


### PREDICTIONS ###
forecast_df = pd.DataFrame()
df['nan'] = np.nan


### Predict trend
trend_2, pred_trend, adjusted_pred_trend = predictTrend(df['trend'][start_index_trend:],
                                         #model_trend=Ridge(random_state=random_state),
                                         model_trend=SVR(),
                                         trend_degree=trend_degree,
                                         N=N)

df['pred_trend'] = [np.nan for _ in range(start_index_trend)] + list(trend_2)
forecast_df['trend'] = list(df['nan'])+list(adjusted_pred_trend)
forecast_df['bad_trend'] = list(df['nan'])+list(pred_trend)


if SHOW:
    df['y'].plot()
    df['trend'].plot()
    df['pred_trend'].plot()
    forecast_df['trend'].plot()
    plt.legend()
    plt.title("Trend forecasting")
    plt.savefig("trend forecasting.png")
    plt.show()



### Predict variance
var_2, pred_var, adjusted_pred_var = predictTrend(df['var'][start_index_var:],
                                         model_trend=Ridge(random_state=random_state),
                                         trend_degree=var_degree,
                                         N=N)

df['pred_var'] = [np.nan for _ in range(start_index_var)] + list(var_2)
forecast_df['var'] = list(df['nan'])+list(adjusted_pred_var)
forecast_df['bad_var'] = list(df['nan'])+list(pred_var)

if SHOW:
    df['y'].plot()
    df['var'].plot()
    df['pred_var'].plot()
    forecast_df['var'].plot()
    plt.legend()
    plt.title("Variance forecasting")
    plt.savefig("Var forecasting.png")
    plt.show()





### Predict seasonnalities
seasonnalities, pred_seasonnalities = extractSeasonnalities(df['normalized'][start_index_seasonnal:],
                                                            model_seasonnal=RandomForestRegressor(n_estimators=30, random_state=random_state),
                                                            periods=periods,
                                                            N=N)

df['seasonnal'] = [0 for _ in range(len(df['y']))]
forecast_df['seasonnal'] = list(df['nan']) + [0 for _ in range(N)]

for period in seasonnalities:
    df['season_'+str(period)] = [np.nan for i in range(start_index_seasonnal)] + list(seasonnalities[period])
    df['seasonnal'] += [np.nan for i in range(start_index_seasonnal)] + list(seasonnalities[period])    
    forecast_df['seasonnal_'+period] = list(df['nan']) + list(pred_seasonnalities[period])
    forecast_df['seasonnal'] += list(df['nan'])+list(pred_seasonnalities[period])

df['seasonnal_error'] = df['normalized']-df['seasonnal']

if SHOW:
    df['normalized'].plot()
    df['seasonnal'].plot()
    plt.legend()
    plt.title("Periodicity forecasting")
    plt.savefig("Periodicity forecasting.png")
    plt.show()




### FINAL PREDICTION ###
df['pred'] = df['seasonnal']*df['var'] + df['trend']
forecast_df['pred'] = forecast_df['seasonnal']*forecast_df['var'] + forecast_df['trend']

df['y'].plot()
#df['pred'].plot()
forecast_df['pred'].plot()
plt.legend()
plt.title(TITLE)
plt.savefig(FILE)
plt.show()

