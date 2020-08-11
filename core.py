import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az
from matplotlib import pyplot as plt
from covid.models.generative import GenerativeModel
from covid.data import summarize_inference_data
import os
import re
import pickle
from datetime import datetime
from covid.data import get_and_process_covidtracking_data, summarize_inference_data

def pickle_object(filename, obj):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile)
    outfile.close()
    return None


def unpickle_object(filename):
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    infile.close()
    return obj


def get_max_date_from_list(datelist):
    dates = []
    for date in datelist:
        try:
            dates.append(datetime.strptime(date, '%d%m%Y'))
        except ValueError:
            pass
    try:
        return max(dates).strftime('%d%m%Y')
    except ValueError:
        return ''


def find_latest_filename(prefix, file_location='/Users/sayantansarkar/Output/covid19india/input/',extension='csv'):
    files = [f for f in os.listdir(file_location) if re.match(prefix+'.+'+extension, f)]
    dates = [re.findall('[0-9]+', filename) for filename in files]
    dates = [date for datelist in dates for date in datelist]
    try:
        max_date = get_max_date_from_list(dates)
        return file_location + prefix + '_' + max_date + '.' + extension
    except ValueError:
        return None


district_hist_data = pd.read_pickle(find_latest_filename(prefix='district_hist_data', extension='pkl'))
buffer_days = 10
city_codes = set(district_hist_data['city_code'])
for city_code in city_codes:
    dt = district_hist_data[district_hist_data['city_code'] == city_code]
    dt['date'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), dt['date']))
    dt = dt[dt['date'] > datetime(2020, 5, 15)]
    dt_state = dt.groupby(['date']).agg({'confirmed': 'sum', 'tested' : 'sum'})
    dt_state.columns = ['positive', 'total']
    new_index = pd.date_range(start=dt_state.index[0] - pd.Timedelta(days=buffer_days), end=dt_state.index[-1], freq="D",)
    dt_state = dt_state.reindex(new_index, fill_value=0)
    try:
        gm = GenerativeModel(city_code, dt_state)
        gm.sample()
        pickle_object(filename='/Users/sayantansarkar/Output/insta_model/'+ city_code + '_gm.pkl', obj=gm)
        result = summarize_inference_data(gm.inference_data)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.set_title(city_code + f" $R_t$")
        samples = gm.trace['r_t']
        x = result.index
        cmap = plt.get_cmap("Reds")
        percs = np.linspace(51, 99, 40)
        colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
        samples = samples.T

        result["median"].plot(c="k", ls='-')

        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(samples, p, axis=1)
            lower = np.percentile(samples, 100 - p, axis=1)
            color_val = colors[i]
            ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=.8)

        ax.axhline(1.0, c="k", lw=1, linestyle="--")
        fig.set_facecolor('w')
        plt.savefig('/Users/sayantansarkar/Output/insta_model/' + city_code + '_rt_plot.png')
        plt.close('all')
        print('Success for '+city_code)
    except Exception as e:
        print('Error for '+city_code)
        print(e)
        continue




#df = get_and_process_covidtracking_data(run_date=pd.Timestamp.today()-pd.Timedelta(days=1))
#region = "OR"
#model_data = df.loc[region]


#gm = GenerativeModel('KA', dt_state)
#gm.sample()
#pickle_object(filename='/Users/sayantansarkar/Output/insta_model/KA_gm_sample.pkl', obj=gm)

# gm = unpickle_object('/Users/sayantansarkar/Output/insta_model/KA_gm_sample.pkl')
# result = summarize_inference_data(gm.inference_data)
# print('lolol')
#
# fig, ax = plt.subplots(figsize=(10,5))
# result.test_adjusted_positive.plot(c="g", label="Test-adjusted")
# result.test_adjusted_positive_raw.plot(c="g", alpha=.5, label="Test-adjusted (raw)", style="--")
# result.infections.plot(c="b", label="Infections")
# gm.observed.positive.plot(c='r', alpha=.7, label="Reported Positives")
# fig.set_facecolor('w')
# ax.legend();
# plt.savefig('/Users/sayantansarkar/Output/insta_model/KA_plot_1.png')
#
# fig, ax = plt.subplots(figsize=(10,5))
#
# ax.set_title(f"Karnataka $R_t$")
# samples = gm.trace['r_t']
# x=result.index
# cmap = plt.get_cmap("Reds")
# percs = np.linspace(51, 99, 40)
# colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
# samples = samples.T
#
# result["median"].plot(c="k", ls='-')
#
# for i, p in enumerate(percs[::-1]):
#     upper = np.percentile(samples, p, axis=1)
#     lower = np.percentile(samples, 100-p, axis=1)
#     color_val = colors[i]
#     ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=.8)
#
# ax.axhline(1.0, c="k", lw=1, linestyle="--")
# fig.set_facecolor('w')
# plt.savefig('/Users/sayantansarkar/Output/insta_model/KA_plot_2.png')