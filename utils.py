import os
import numpy as np
import pandas as pd

def create_features(data):
    """
    Create features
    """

    # Seasonality
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.isocalendar().week

    # Avoid the UInt32 because STAN can not handle it
    data['WeekOfYear'] = data['WeekOfYear'].astype(np.int32)

    # Competition
    data['CompetitionOpen'] = (12 * (data.Year - data.CompetitionOpenSinceYear) 
        + (data.Month - data.CompetitionOpenSinceMonth))
    data['PromoOpen'] = (12 * (data.Year - data.Promo2SinceYear) + 
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)        
    
    # Promotion
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & 
                (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    cols_to_drop = ['monthStr', 'PromoInterval']
    data = data.drop(cols_to_drop, axis=1)

    # Dummies
    #dummy_cols = ['StateHoliday', 'StoreType', 'Assortment']
    #data = pd.get_dummies(data, columns=dummy_cols)

    return data


def create_holidays(data):
    """
    Create holidays used in Prophet
    """

    school_holidays = data[data.SchoolHoliday == 1]['Date'].values
    school_holidays = np.unique(school_holidays)
    school_holidays = pd.DataFrame({'ds':school_holidays, 
                                    'holiday':'school_holiday'})

    mask = ((data.StateHoliday == 'a') | (data.StateHoliday == 'b') | 
            (data.StateHoliday == 'c'))
    state_holidays = data[mask]['Date'].values
    state_holidays = np.unique(state_holidays)
    state_holidays = pd.DataFrame({'ds':state_holidays, 
                                    'holiday':'state_holiday'})

    return pd.concat([school_holidays, state_holidays])


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


# define rmspe for xgb 
# (code from https://www.kaggle.com/cast42/xgboost-in-python-with-rmspe-v2/code)
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)

