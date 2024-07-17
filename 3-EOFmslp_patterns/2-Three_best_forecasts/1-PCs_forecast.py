# %% [markdown]
# # 1. Principal Component Forecast

# This script is used to use seasonal forecasts systems to predict NAO, EA, EAWR and SCA. 

#%%
print("1. Principal Component Forecast")

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import pandas as pd
import numpy as np
import locale
import calendar
import matplotlib
import matplotlib.pyplot as plt
import xskillscore as xs
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) > 2:
    aggr = str(sys.argv[1])
    startmonth = int(sys.argv[2])
# If no variables were introduced, ask for them
else:
    # Subset of plots to be produced
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m ]: ")

    # Which start month
    startmonth = int(input("Mes de inicialización (en número): "))

    # # Forecast month
    # if aggr=='1m':
    #     answ = input("Resultados para el mes [ Jan , Feb , Mar , Apr , May , Jun , Jul , Aug , Sep , Oct , Nov , Dec ]: ")
    #     endmonth = np.where(np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']) == answ)[0][0]+1
    # else:
    #     answ = input("Resultados para el trimestre [ NDJ , DJF , JFM , FMA , MAM , AMJ , MJJ , JJA , JAS , ASO , SON , OND ]: ")
    #     endmonth = np.where(np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']) == answ)[0][0]+1

# Dictionary to link full system names and simplier names
full_name = {'ECMWF-System 4': ['ecmwf','4'],
             'ECMWF-SEAS5': ['ecmwf', '5'],
             'ECMWF-SEAS5.1': ['ecmwf', '51'],
             'Météo France-System 5': [ 'meteo_france', '5'],
             'Météo France-System 6': [ 'meteo_france', '6'],
             'Météo France-System 7': [ 'meteo_france', '7'],
             'Météo France-System 8': [ 'meteo_france', '8'],
             'Met Office-System 12': ['ukmo', '12'],
             'Met Office-System 13': ['ukmo', '13'],
             'Met Office-System 14': ['ukmo', '14'],
             'Met Office-System 15': ['ukmo', '15'],
             'Met Office-GloSea6': ['ukmo', '600'],
             'Met Office-GloSea6.1': ['ukmo', '601'],
             'Met Office-GloSea6.2': ['ukmo', '602'],
             'Met Office-GloSea6.3': ['ukmo', '603'],
             'DWD-GCFS2.0': ['dwd', '2'],
             'DWD-GCFS2.1': ['dwd', '21'],
             'CMCC-SPSv3.0': ['cmcc', '3'],
             'CMCC-SPSv3.5': ['cmcc', '35'],
             'NCEP-CFSv2': ['ncep', '2'],
             'JMA-CPS2': ['jma', '2'],
             'JMA-CPS3': ['jma', '3'],
             'ECCC-GEM-NEMO': ['eccc', '1'],
             'ECCC-CanCM4i': ['eccc', '2'],
             'ECCC-GEM5-NEMO': ['eccc', '3']
            }

# Number of lead-times
lts=3
# List of initializations
if aggr=='1m':
    endmonth_name = np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    initialization = {endmonth_name[startmonth+(l+1)-1 if startmonth+(l+1)-1<12 else startmonth+(l+1)-1-12]: [startmonth+(l+1) if startmonth+(l+1)<=12 else startmonth+(l+1)-12, l+2] for l in reversed(range(lts))}
elif aggr=='3m': 
    endmonth_name = np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND'])
    initialization = {endmonth_name[startmonth+(l+3)-1 if startmonth+(l+3)-1<12 else startmonth+(l+3)-1-12]: [startmonth+(l+3) if startmonth+(l+3)<=12 else startmonth+(l+3)-12, l+4] for l in reversed(range(lts))}

config = dict(
    hcstarty = 1993,
    hcendy = 2016,
    aggr = aggr,
)


# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
MODESDATADIR = os.getenv('MODES_DIR')
MODESDIR = MODESDATADIR + '/modes'
CSVDIR = os.getenv('CSV_DIR')

# EOF1
eof1_rpss = pd.read_csv(CSVDIR+'/Score-card_rpss_'+aggr+'_eof1.csv')
eof1_rpss = eof1_rpss.rename(columns={eof1_rpss.columns[0]: "Model", eof1_rpss.columns[1]: "lead"})
eof1_rpss_lead3 = eof1_rpss[["Model","lead",list(initialization)[-3]]]
eof1_rpss_lead3 = eof1_rpss_lead3[eof1_rpss_lead3['lead']=='lead3'].sort_values(by=list(initialization)[-3], ascending=False)
eof1_rpss_lead2 = eof1_rpss[["Model","lead",list(initialization)[-2]]]
eof1_rpss_lead2 = eof1_rpss_lead2[eof1_rpss_lead2['lead']=='lead2'].sort_values(by=list(initialization)[-2], ascending=False)
eof1_rpss_lead1 = eof1_rpss[["Model","lead",list(initialization)[-1]]]
eof1_rpss_lead1 = eof1_rpss_lead1[eof1_rpss_lead1['lead']=='lead1'].sort_values(by=list(initialization)[-1], ascending=False)
# EOF1 - lead3
origin = full_name[eof1_rpss_lead3['Model'].values[0]][0]
system = full_name[eof1_rpss_lead3['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-3]][1]
hcpcs_eof1_lead3 = xr.open_dataset(hcpcs_fname).sel(mode=0,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof1_lead3.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof1_lead3.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof1_lead3_low = (hcpcs_eof1_lead3.where(hcpcs_eof1_lead3<low).count(dim='number')/float(hcpcs_eof1_lead3.number.size)).assign_coords({'category':-1})
hcpcs_eof1_lead3_med = (hcpcs_eof1_lead3.where((hcpcs_eof1_lead3<=high) & (hcpcs_eof1_lead3>=low)).count(dim='number')/float(hcpcs_eof1_lead3.number.size)).assign_coords({'category':0})
hcpcs_eof1_lead3_high = (hcpcs_eof1_lead3.where(hcpcs_eof1_lead3>high).count(dim='number')/float(hcpcs_eof1_lead3.number.size)).assign_coords({'category':1})
hcpcs_eof1_lead3_ter = xr.concat([hcpcs_eof1_lead3_low,hcpcs_eof1_lead3_med,hcpcs_eof1_lead3_high],dim='category')
hcpcs_eof1_lead3_val = hcpcs_eof1_lead3_ter.idxmax(dim='category').drop('mode')
hcpcs_eof1_lead3_val = hcpcs_eof1_lead3_val.assign_coords({'rpss':eof1_rpss_lead3[list(initialization)[-3]].values[0]})
# EOF1 - lead2
origin = full_name[eof1_rpss_lead2['Model'].values[0]][0]
system = full_name[eof1_rpss_lead2['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-2]][1]
hcpcs_eof1_lead2 = xr.open_dataset(hcpcs_fname).sel(mode=0,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof1_lead2.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof1_lead2.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof1_lead2_low = (hcpcs_eof1_lead2.where(hcpcs_eof1_lead2<low).count(dim='number')/float(hcpcs_eof1_lead2.number.size)).assign_coords({'category':-1})
hcpcs_eof1_lead2_med = (hcpcs_eof1_lead2.where((hcpcs_eof1_lead2<=high) & (hcpcs_eof1_lead2>=low)).count(dim='number')/float(hcpcs_eof1_lead2.number.size)).assign_coords({'category':0})
hcpcs_eof1_lead2_high = (hcpcs_eof1_lead2.where(hcpcs_eof1_lead2>high).count(dim='number')/float(hcpcs_eof1_lead2.number.size)).assign_coords({'category':1})
hcpcs_eof1_lead2_ter = xr.concat([hcpcs_eof1_lead2_low,hcpcs_eof1_lead2_med,hcpcs_eof1_lead2_high],dim='category')
hcpcs_eof1_lead2_val = hcpcs_eof1_lead2_ter.idxmax(dim='category').drop('mode')
hcpcs_eof1_lead2_val = hcpcs_eof1_lead2_val.assign_coords({'rpss':eof1_rpss_lead2[list(initialization)[-2]].values[0]})
# EOF1 - lead1
origin = full_name[eof1_rpss_lead1['Model'].values[0]][0]
system = full_name[eof1_rpss_lead1['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-1]][1]
hcpcs_eof1_lead1 = xr.open_dataset(hcpcs_fname).sel(mode=0,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof1_lead1.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof1_lead1.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof1_lead1_low = (hcpcs_eof1_lead1.where(hcpcs_eof1_lead1<low).count(dim='number')/float(hcpcs_eof1_lead1.number.size)).assign_coords({'category':-1})
hcpcs_eof1_lead1_med = (hcpcs_eof1_lead1.where((hcpcs_eof1_lead1<=high) & (hcpcs_eof1_lead1>=low)).count(dim='number')/float(hcpcs_eof1_lead1.number.size)).assign_coords({'category':0})
hcpcs_eof1_lead1_high = (hcpcs_eof1_lead1.where(hcpcs_eof1_lead1>high).count(dim='number')/float(hcpcs_eof1_lead1.number.size)).assign_coords({'category':1})
hcpcs_eof1_lead1_ter = xr.concat([hcpcs_eof1_lead1_low,hcpcs_eof1_lead1_med,hcpcs_eof1_lead1_high],dim='category')
hcpcs_eof1_lead1_val = hcpcs_eof1_lead1_ter.idxmax(dim='category').drop('mode')
hcpcs_eof1_lead1_val = hcpcs_eof1_lead1_val.assign_coords({'rpss':eof1_rpss_lead1[list(initialization)[-1]].values[0]})

# EOF2
eof2_rpss = pd.read_csv(CSVDIR+'/Score-card_rpss_'+aggr+'_eof2.csv')
eof2_rpss = eof2_rpss.rename(columns={eof2_rpss.columns[0]: "Model", eof2_rpss.columns[1]: "lead"})
eof2_rpss_lead3 = eof2_rpss[["Model","lead",list(initialization)[-3]]]
eof2_rpss_lead3 = eof2_rpss_lead3[eof2_rpss_lead3['lead']=='lead3'].sort_values(by=list(initialization)[-3], ascending=False)
eof2_rpss_lead2 = eof2_rpss[["Model","lead",list(initialization)[-2]]]
eof2_rpss_lead2 = eof2_rpss_lead2[eof2_rpss_lead2['lead']=='lead2'].sort_values(by=list(initialization)[-2], ascending=False)
eof2_rpss_lead1 = eof2_rpss[["Model","lead",list(initialization)[-1]]]
eof2_rpss_lead1 = eof2_rpss_lead1[eof2_rpss_lead1['lead']=='lead1'].sort_values(by=list(initialization)[-1], ascending=False)
# EOF2 - lead3
origin = full_name[eof2_rpss_lead3['Model'].values[0]][0]
system = full_name[eof2_rpss_lead3['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-3]][1]
hcpcs_eof2_lead3 = xr.open_dataset(hcpcs_fname).sel(mode=1,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof2_lead3.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof2_lead3.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof2_lead3_low = (hcpcs_eof2_lead3.where(hcpcs_eof2_lead3<low).count(dim='number')/float(hcpcs_eof2_lead3.number.size)).assign_coords({'category':-1})
hcpcs_eof2_lead3_med = (hcpcs_eof2_lead3.where((hcpcs_eof2_lead3<=high) & (hcpcs_eof2_lead3>=low)).count(dim='number')/float(hcpcs_eof2_lead3.number.size)).assign_coords({'category':0})
hcpcs_eof2_lead3_high = (hcpcs_eof2_lead3.where(hcpcs_eof2_lead3>high).count(dim='number')/float(hcpcs_eof2_lead3.number.size)).assign_coords({'category':1})
hcpcs_eof2_lead3_ter = xr.concat([hcpcs_eof2_lead3_low,hcpcs_eof2_lead3_med,hcpcs_eof2_lead3_high],dim='category')
hcpcs_eof2_lead3_val = hcpcs_eof2_lead3_ter.idxmax(dim='category').drop('mode')
hcpcs_eof2_lead3_val = hcpcs_eof2_lead3_val.assign_coords({'rpss':eof2_rpss_lead3[list(initialization)[-3]].values[0]})
# EOF2 - lead2
origin = full_name[eof2_rpss_lead2['Model'].values[0]][0]
system = full_name[eof2_rpss_lead2['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-2]][1]
hcpcs_eof2_lead2 = xr.open_dataset(hcpcs_fname).sel(mode=1,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof2_lead2.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof2_lead2.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof2_lead2_low = (hcpcs_eof2_lead2.where(hcpcs_eof2_lead2<low).count(dim='number')/float(hcpcs_eof2_lead2.number.size)).assign_coords({'category':-1})
hcpcs_eof2_lead2_med = (hcpcs_eof2_lead2.where((hcpcs_eof2_lead2<=high) & (hcpcs_eof2_lead2>=low)).count(dim='number')/float(hcpcs_eof2_lead2.number.size)).assign_coords({'category':0})
hcpcs_eof2_lead2_high = (hcpcs_eof2_lead2.where(hcpcs_eof2_lead2>high).count(dim='number')/float(hcpcs_eof2_lead2.number.size)).assign_coords({'category':1})
hcpcs_eof2_lead2_ter = xr.concat([hcpcs_eof2_lead2_low,hcpcs_eof2_lead2_med,hcpcs_eof2_lead2_high],dim='category')
hcpcs_eof2_lead2_val = hcpcs_eof2_lead2_ter.idxmax(dim='category').drop('mode')
hcpcs_eof2_lead2_val = hcpcs_eof2_lead2_val.assign_coords({'rpss':eof2_rpss_lead2[list(initialization)[-2]].values[0]})
# EOF2 - lead1
origin = full_name[eof2_rpss_lead1['Model'].values[0]][0]
system = full_name[eof2_rpss_lead1['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-1]][1]
hcpcs_eof2_lead1 = xr.open_dataset(hcpcs_fname).sel(mode=1,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof2_lead1.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof2_lead1.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof2_lead1_low = (hcpcs_eof2_lead1.where(hcpcs_eof2_lead1<low).count(dim='number')/float(hcpcs_eof2_lead1.number.size)).assign_coords({'category':-1})
hcpcs_eof2_lead1_med = (hcpcs_eof2_lead1.where((hcpcs_eof2_lead1<=high) & (hcpcs_eof2_lead1>=low)).count(dim='number')/float(hcpcs_eof2_lead1.number.size)).assign_coords({'category':0})
hcpcs_eof2_lead1_high = (hcpcs_eof2_lead1.where(hcpcs_eof2_lead1>high).count(dim='number')/float(hcpcs_eof2_lead1.number.size)).assign_coords({'category':1})
hcpcs_eof2_lead1_ter = xr.concat([hcpcs_eof2_lead1_low,hcpcs_eof2_lead1_med,hcpcs_eof2_lead1_high],dim='category')
hcpcs_eof2_lead1_val = hcpcs_eof2_lead1_ter.idxmax(dim='category').drop('mode')
hcpcs_eof2_lead1_val = hcpcs_eof2_lead1_val.assign_coords({'rpss':eof2_rpss_lead1[list(initialization)[-1]].values[0]})

# EOF3
eof3_rpss = pd.read_csv(CSVDIR+'/Score-card_rpss_'+aggr+'_eof3.csv')
eof3_rpss = eof3_rpss.rename(columns={eof3_rpss.columns[0]: "Model", eof3_rpss.columns[1]: "lead"})
eof3_rpss_lead3 = eof3_rpss[["Model","lead",list(initialization)[-3]]]
eof3_rpss_lead3 = eof3_rpss_lead3[eof3_rpss_lead3['lead']=='lead3'].sort_values(by=list(initialization)[-3], ascending=False)
eof3_rpss_lead2 = eof3_rpss[["Model","lead",list(initialization)[-2]]]
eof3_rpss_lead2 = eof3_rpss_lead2[eof3_rpss_lead2['lead']=='lead2'].sort_values(by=list(initialization)[-2], ascending=False)
eof3_rpss_lead1 = eof3_rpss[["Model","lead",list(initialization)[-1]]]
eof3_rpss_lead1 = eof3_rpss_lead1[eof3_rpss_lead1['lead']=='lead1'].sort_values(by=list(initialization)[-1], ascending=False)
# EOF3 - lead3
origin = full_name[eof3_rpss_lead3['Model'].values[0]][0]
system = full_name[eof3_rpss_lead3['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-3]][1]
hcpcs_eof3_lead3 = xr.open_dataset(hcpcs_fname).sel(mode=2,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof3_lead3.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof3_lead3.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof3_lead3_low = (hcpcs_eof3_lead3.where(hcpcs_eof3_lead3<low).count(dim='number')/float(hcpcs_eof3_lead3.number.size)).assign_coords({'category':-1})
hcpcs_eof3_lead3_med = (hcpcs_eof3_lead3.where((hcpcs_eof3_lead3<=high) & (hcpcs_eof3_lead3>=low)).count(dim='number')/float(hcpcs_eof3_lead3.number.size)).assign_coords({'category':0})
hcpcs_eof3_lead3_high = (hcpcs_eof3_lead3.where(hcpcs_eof3_lead3>high).count(dim='number')/float(hcpcs_eof3_lead3.number.size)).assign_coords({'category':1})
hcpcs_eof3_lead3_ter = xr.concat([hcpcs_eof3_lead3_low,hcpcs_eof3_lead3_med,hcpcs_eof3_lead3_high],dim='category')
hcpcs_eof3_lead3_val = hcpcs_eof3_lead3_ter.idxmax(dim='category').drop('mode')
hcpcs_eof3_lead3_val = hcpcs_eof3_lead3_val.assign_coords({'rpss':eof3_rpss_lead3[list(initialization)[-3]].values[0]})
# EOF3 - lead2
origin = full_name[eof3_rpss_lead2['Model'].values[0]][0]
system = full_name[eof3_rpss_lead2['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-2]][1]
hcpcs_eof3_lead2 = xr.open_dataset(hcpcs_fname).sel(mode=2,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof3_lead2.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof3_lead2.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof3_lead2_low = (hcpcs_eof3_lead2.where(hcpcs_eof3_lead2<low).count(dim='number')/float(hcpcs_eof3_lead2.number.size)).assign_coords({'category':-1})
hcpcs_eof3_lead2_med = (hcpcs_eof3_lead2.where((hcpcs_eof3_lead2<=high) & (hcpcs_eof3_lead2>=low)).count(dim='number')/float(hcpcs_eof3_lead2.number.size)).assign_coords({'category':0})
hcpcs_eof3_lead2_high = (hcpcs_eof3_lead2.where(hcpcs_eof3_lead2>high).count(dim='number')/float(hcpcs_eof3_lead2.number.size)).assign_coords({'category':1})
hcpcs_eof3_lead2_ter = xr.concat([hcpcs_eof3_lead2_low,hcpcs_eof3_lead2_med,hcpcs_eof3_lead2_high],dim='category')
hcpcs_eof3_lead2_val = hcpcs_eof3_lead2_ter.idxmax(dim='category').drop('mode')
hcpcs_eof3_lead2_val = hcpcs_eof3_lead2_val.assign_coords({'rpss':eof3_rpss_lead2[list(initialization)[-2]].values[0]})
# EOF3 - lead1
origin = full_name[eof3_rpss_lead1['Model'].values[0]][0]
system = full_name[eof3_rpss_lead1['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-1]][1]
hcpcs_eof3_lead1 = xr.open_dataset(hcpcs_fname).sel(mode=2,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof3_lead1.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof3_lead1.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof3_lead1_low = (hcpcs_eof3_lead1.where(hcpcs_eof3_lead1<low).count(dim='number')/float(hcpcs_eof3_lead1.number.size)).assign_coords({'category':-1})
hcpcs_eof3_lead1_med = (hcpcs_eof3_lead1.where((hcpcs_eof3_lead1<=high) & (hcpcs_eof3_lead1>=low)).count(dim='number')/float(hcpcs_eof3_lead1.number.size)).assign_coords({'category':0})
hcpcs_eof3_lead1_high = (hcpcs_eof3_lead1.where(hcpcs_eof3_lead1>high).count(dim='number')/float(hcpcs_eof3_lead1.number.size)).assign_coords({'category':1})
hcpcs_eof3_lead1_ter = xr.concat([hcpcs_eof3_lead1_low,hcpcs_eof3_lead1_med,hcpcs_eof3_lead1_high],dim='category')
hcpcs_eof3_lead1_val = hcpcs_eof3_lead1_ter.idxmax(dim='category').drop('mode')
hcpcs_eof3_lead1_val = hcpcs_eof3_lead1_val.assign_coords({'rpss':eof3_rpss_lead1[list(initialization)[-1]].values[0]})

# EOF4
eof4_rpss = pd.read_csv(CSVDIR+'/Score-card_rpss_'+aggr+'_eof4.csv')
eof4_rpss = eof4_rpss.rename(columns={eof4_rpss.columns[0]: "Model", eof4_rpss.columns[1]: "lead"})
eof4_rpss_lead3 = eof4_rpss[["Model","lead",list(initialization)[-3]]]
eof4_rpss_lead3 = eof4_rpss_lead3[eof4_rpss_lead3['lead']=='lead3'].sort_values(by=list(initialization)[-3], ascending=False)
eof4_rpss_lead2 = eof4_rpss[["Model","lead",list(initialization)[-2]]]
eof4_rpss_lead2 = eof4_rpss_lead2[eof4_rpss_lead2['lead']=='lead2'].sort_values(by=list(initialization)[-2], ascending=False)
eof4_rpss_lead1 = eof4_rpss[["Model","lead",list(initialization)[-1]]]
eof4_rpss_lead1 = eof4_rpss_lead1[eof4_rpss_lead1['lead']=='lead1'].sort_values(by=list(initialization)[-1], ascending=False)
# EOF4 - lead3
origin = full_name[eof4_rpss_lead3['Model'].values[0]][0]
system = full_name[eof4_rpss_lead3['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-3]][1]
hcpcs_eof4_lead3 = xr.open_dataset(hcpcs_fname).sel(mode=3,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof4_lead3.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof4_lead3.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof4_lead3_low = (hcpcs_eof4_lead3.where(hcpcs_eof4_lead3<low).count(dim='number')/float(hcpcs_eof4_lead3.number.size)).assign_coords({'category':-1})
hcpcs_eof4_lead3_med = (hcpcs_eof4_lead3.where((hcpcs_eof4_lead3<=high) & (hcpcs_eof4_lead3>=low)).count(dim='number')/float(hcpcs_eof4_lead3.number.size)).assign_coords({'category':0})
hcpcs_eof4_lead3_high = (hcpcs_eof4_lead3.where(hcpcs_eof4_lead3>high).count(dim='number')/float(hcpcs_eof4_lead3.number.size)).assign_coords({'category':1})
hcpcs_eof4_lead3_ter = xr.concat([hcpcs_eof4_lead3_low,hcpcs_eof4_lead3_med,hcpcs_eof4_lead3_high],dim='category')
hcpcs_eof4_lead3_val = hcpcs_eof4_lead3_ter.idxmax(dim='category').drop('mode')
hcpcs_eof4_lead3_val = hcpcs_eof4_lead3_val.assign_coords({'rpss':eof4_rpss_lead3[list(initialization)[-3]].values[0]})
# EOF4 - lead2
origin = full_name[eof4_rpss_lead2['Model'].values[0]][0]
system = full_name[eof4_rpss_lead2['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-2]][1]
hcpcs_eof4_lead2 = xr.open_dataset(hcpcs_fname).sel(mode=3,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof4_lead2.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof4_lead2.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof4_lead2_low = (hcpcs_eof4_lead2.where(hcpcs_eof4_lead2<low).count(dim='number')/float(hcpcs_eof4_lead2.number.size)).assign_coords({'category':-1})
hcpcs_eof4_lead2_med = (hcpcs_eof4_lead2.where((hcpcs_eof4_lead2<=high) & (hcpcs_eof4_lead2>=low)).count(dim='number')/float(hcpcs_eof4_lead2.number.size)).assign_coords({'category':0})
hcpcs_eof4_lead2_high = (hcpcs_eof4_lead2.where(hcpcs_eof4_lead2>high).count(dim='number')/float(hcpcs_eof4_lead2.number.size)).assign_coords({'category':1})
hcpcs_eof4_lead2_ter = xr.concat([hcpcs_eof4_lead2_low,hcpcs_eof4_lead2_med,hcpcs_eof4_lead2_high],dim='category')
hcpcs_eof4_lead2_val = hcpcs_eof4_lead2_ter.idxmax(dim='category').drop('mode')
hcpcs_eof4_lead2_val = hcpcs_eof4_lead2_val.assign_coords({'rpss':eof4_rpss_lead2[list(initialization)[-2]].values[0]})
# EOF4 - lead1
origin = full_name[eof4_rpss_lead1['Model'].values[0]][0]
system = full_name[eof4_rpss_lead1['Model'].values[0]][1]
# Reading HCST data from file
hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
fcmonth = initialization[list(initialization)[-1]][1]
hcpcs_eof4_lead1 = xr.open_dataset(hcpcs_fname).sel(mode=3,forecastMonth=fcmonth)
# Obtain forecast
low = hcpcs_eof4_lead1.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
high = hcpcs_eof4_lead1.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
hcpcs_eof4_lead1_low = (hcpcs_eof4_lead1.where(hcpcs_eof4_lead1<low).count(dim='number')/float(hcpcs_eof4_lead1.number.size)).assign_coords({'category':-1})
hcpcs_eof4_lead1_med = (hcpcs_eof4_lead1.where((hcpcs_eof4_lead1<=high) & (hcpcs_eof4_lead1>=low)).count(dim='number')/float(hcpcs_eof4_lead1.number.size)).assign_coords({'category':0})
hcpcs_eof4_lead1_high = (hcpcs_eof4_lead1.where(hcpcs_eof4_lead1>high).count(dim='number')/float(hcpcs_eof4_lead1.number.size)).assign_coords({'category':1})
hcpcs_eof4_lead1_ter = xr.concat([hcpcs_eof4_lead1_low,hcpcs_eof4_lead1_med,hcpcs_eof4_lead1_high],dim='category')
hcpcs_eof4_lead1_val = hcpcs_eof4_lead1_ter.idxmax(dim='category').drop('mode')
hcpcs_eof4_lead1_val = hcpcs_eof4_lead1_val.assign_coords({'rpss':eof4_rpss_lead1[list(initialization)[-1]].values[0]})

hcpcs_eof1_val = xr.concat([hcpcs_eof1_lead1_val,hcpcs_eof1_lead2_val,hcpcs_eof1_lead3_val],dim='forecastMonth').assign_coords({'mode':0})
hcpcs_eof2_val = xr.concat([hcpcs_eof2_lead1_val,hcpcs_eof2_lead2_val,hcpcs_eof2_lead3_val],dim='forecastMonth').assign_coords({'mode':1})
hcpcs_eof3_val = xr.concat([hcpcs_eof3_lead1_val,hcpcs_eof3_lead2_val,hcpcs_eof3_lead3_val],dim='forecastMonth').assign_coords({'mode':2})
hcpcs_eof4_val = xr.concat([hcpcs_eof4_lead1_val,hcpcs_eof4_lead2_val,hcpcs_eof4_lead3_val],dim='forecastMonth').assign_coords({'mode':3})
hcpcs_val = xr.concat([hcpcs_eof1_val,hcpcs_eof2_val,hcpcs_eof3_val,hcpcs_eof4_val],dim='mode')
hcpcs_val['valid_time'] = hcpcs_eof1_val['valid_time']

# Directory selection
POSTDIR = os.getenv('POST_DIR')
NEWSCOREDIR = POSTDIR + '/scores'
NEWMODESDIR = POSTDIR + '/modes'
NEWPLOTSDIR = f'./plots/PCS'
# Directory creation
for directory in [NEWMODESDIR, NEWSCOREDIR, NEWPLOTSDIR]:
    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

hcpcs_val.to_netcdf(f'{NEWMODESDIR}/PCs_best_forecasts_stmonth{startmonth:02d}.'+aggr+'.nc')





# # Base name for observations
# obs_bname = 'era5_monthly_stmonth{start_month:02d}_{hcstarty}-{hcendy}'.format(start_month=initialization[list(initialization)[-1]][0],**config)
# # File name for observations
# obpcs_fname = f'{MODESDIR}/{obs_bname}.{aggr}.PCs.nc'
# # Reading OBS data from file
# obpcs = xr.open_dataset(obpcs_fname)

# fig = plt.figure(figsize=(12,12))
# gs = fig.add_gridspec(4,1)
# ax1 = fig.add_subplot(gs[0,0])
# thispc_obs = obpcs.where(obpcs.valid_time==hcpcs_eof1_lead1_val.valid_time,drop=True).pseudo_pcs.sel(mode=0)
# years = thispc_obs.valid_time.dt.year
# plt.plot(years, thispc_obs, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc_obs.squeeze(), where=(thispc_obs.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc_obs.squeeze(), 0, where=(thispc_obs.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('EOF1')
# plt.ylim([-2., 2.])
# ax1.set_xticks(years.values[::2])    
# ax1.set_xticklabels(years.values[::2])    
# ax1.grid(True)
# ax2 = fig.add_subplot(gs[1,0])
# thispc = hcpcs_eof1_lead1_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead1')
# plt.ylim([-2., 2.])
# ax2.set_xticks(years.values[::2])    
# ax2.set_xticklabels(years.values[::2])    
# ax2.grid(True)
# ax3 = fig.add_subplot(gs[2,0])
# thispc = hcpcs_eof1_lead2_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead2')
# plt.ylim([-2., 2.])
# ax3.set_xticks(years.values[::2])    
# ax3.set_xticklabels(years.values[::2])    
# ax3.grid(True)
# ax4 = fig.add_subplot(gs[3,0])
# thispc = hcpcs_eof1_lead3_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead3')
# plt.ylim([-2., 2.])
# ax4.set_xticks(years.values[::2])    
# ax4.set_xticklabels(years.values[::2])    
# ax4.grid(True)
# # Titles
# fig.suptitle(f'Best {endmonth_name} EOF1 forecasts', fontsize=16)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.25, wspace=0.15)
# # Save figure
# figname = f'{NEWMODESDIR}/EOF1_forecasts_{endmonth_name}.png'
# fig.savefig(figname,dpi=600)  


# fig = plt.figure(figsize=(12,12))
# gs = fig.add_gridspec(4,1)
# ax1 = fig.add_subplot(gs[0,0])
# thispc_obs = obpcs.where(obpcs.valid_time==hcpcs_eof2_lead1_val.valid_time,drop=True).pseudo_pcs.sel(mode=1)
# years = thispc_obs.valid_time.dt.year
# plt.plot(years, thispc_obs, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc_obs.squeeze(), where=(thispc_obs.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc_obs.squeeze(), 0, where=(thispc_obs.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('EOF2')
# plt.ylim([-2., 2.])
# ax1.set_xticks(years.values[::2])    
# ax1.set_xticklabels(years.values[::2])    
# ax1.grid(True)
# ax2 = fig.add_subplot(gs[1,0])
# thispc = hcpcs_eof2_lead1_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead1')
# plt.ylim([-2., 2.])
# ax2.set_xticks(years.values[::2])    
# ax2.set_xticklabels(years.values[::2])    
# ax2.grid(True)
# ax3 = fig.add_subplot(gs[2,0])
# thispc = hcpcs_eof2_lead2_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead2')
# plt.ylim([-2., 2.])
# ax3.set_xticks(years.values[::2])    
# ax3.set_xticklabels(years.values[::2])    
# ax3.grid(True)
# ax4 = fig.add_subplot(gs[3,0])
# thispc = hcpcs_eof2_lead3_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead3')
# plt.ylim([-2., 2.])
# ax4.set_xticks(years.values[::2])    
# ax4.set_xticklabels(years.values[::2])    
# ax4.grid(True)
# # Titles
# fig.suptitle(f'Best {endmonth_name} EOF2 forecasts', fontsize=16)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.25, wspace=0.15)
# # Save figure
# figname = f'{NEWMODESDIR}/EOF2_forecasts_{endmonth_name}.png'
# fig.savefig(figname,dpi=600)  

# fig = plt.figure(figsize=(12,12))
# gs = fig.add_gridspec(4,1)
# ax1 = fig.add_subplot(gs[0,0])
# thispc_obs = obpcs.where(obpcs.valid_time==hcpcs_eof3_lead1_val.valid_time,drop=True).pseudo_pcs.sel(mode=2)
# years = thispc_obs.valid_time.dt.year
# plt.plot(years, thispc_obs, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc_obs.squeeze(), where=(thispc_obs.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc_obs.squeeze(), 0, where=(thispc_obs.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('EOF3')
# plt.ylim([-2., 2.])
# ax1.set_xticks(years.values[::2])    
# ax1.set_xticklabels(years.values[::2])    
# ax1.grid(True)
# ax2 = fig.add_subplot(gs[1,0])
# thispc = hcpcs_eof3_lead1_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead1')
# plt.ylim([-2., 2.])
# ax2.set_xticks(years.values[::2])    
# ax2.set_xticklabels(years.values[::2])    
# ax2.grid(True)
# ax3 = fig.add_subplot(gs[2,0])
# thispc = hcpcs_eof3_lead2_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead2')
# plt.ylim([-2., 2.])
# ax3.set_xticks(years.values[::2])    
# ax3.set_xticklabels(years.values[::2])    
# ax3.grid(True)
# ax4 = fig.add_subplot(gs[3,0])
# thispc = hcpcs_eof3_lead3_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead3')
# plt.ylim([-2., 2.])
# ax4.set_xticks(years.values[::2])    
# ax4.set_xticklabels(years.values[::2])    
# ax4.grid(True)
# # Titles
# fig.suptitle(f'Best {endmonth_name} EOF3 forecasts', fontsize=16)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.25, wspace=0.15)
# # Save figure
# figname = f'{NEWMODESDIR}/EOF3_forecasts_{endmonth_name}.png'
# fig.savefig(figname,dpi=600)  

# fig = plt.figure(figsize=(12,12))
# gs = fig.add_gridspec(4,1)
# ax1 = fig.add_subplot(gs[0,0])
# thispc_obs = obpcs.where(obpcs.valid_time==hcpcs_eof4_lead1_val.valid_time,drop=True).pseudo_pcs.sel(mode=3)
# years = thispc_obs.valid_time.dt.year
# plt.plot(years, thispc_obs, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc_obs.squeeze(), where=(thispc_obs.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc_obs.squeeze(), 0, where=(thispc_obs.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('EOF4')
# plt.ylim([-2., 2.])
# ax1.set_xticks(years.values[::2])    
# ax1.set_xticklabels(years.values[::2])    
# ax1.grid(True)
# ax2 = fig.add_subplot(gs[1,0])
# thispc = hcpcs_eof4_lead1_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead1')
# plt.ylim([-2., 2.])
# ax2.set_xticks(years.values[::2])    
# ax2.set_xticklabels(years.values[::2])    
# ax2.grid(True)
# ax3 = fig.add_subplot(gs[2,0])
# thispc = hcpcs_eof4_lead2_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead2')
# plt.ylim([-2., 2.])
# ax3.set_xticks(years.values[::2])    
# ax3.set_xticklabels(years.values[::2])    
# ax3.grid(True)
# ax4 = fig.add_subplot(gs[3,0])
# thispc = hcpcs_eof4_lead3_val.pseudo_pcs
# years = thispc.valid_time.dt.year
# plt.plot(years, thispc, linewidth=2, color='k')
# plt.fill_between(years, 0, thispc.squeeze(), where=(thispc.squeeze() >= 0), color='firebrick',
#                 interpolate=True)
# plt.fill_between(years, thispc.squeeze(), 0, where=(thispc.squeeze() <= 0), color='lightblue',
#                 interpolate=True)
# plt.axhline(0, color='k')
# plt.xlabel('Year')
# plt.ylabel('Forecast lead3')
# plt.ylim([-2., 2.])
# ax4.set_xticks(years.values[::2])    
# ax4.set_xticklabels(years.values[::2])    
# ax4.grid(True)
# # Titles
# fig.suptitle(f'Best {endmonth_name} EOF4 forecasts', fontsize=16)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.25, wspace=0.15)
# # Save figure
# figname = f'{NEWMODESDIR}/EOF4_forecasts_{endmonth_name}.png'
# fig.savefig(figname,dpi=600)  
