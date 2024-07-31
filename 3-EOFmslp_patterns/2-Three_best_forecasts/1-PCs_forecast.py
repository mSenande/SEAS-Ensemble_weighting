# %% [markdown]
# # 1. Principal Component Forecast

# This script uses seasonal forecast systems to predict the four main variability patterns for the hindcast period.

# Information about the skill of each forecast system (RPSS) is used to select the best performing system for each start-month and forecasted-season pair.
# In this way, considering the 3-month aggregation, each forecasted season would have 3 variability pattern forecasts (one for each initialization / leadtime).

# First we have to decide a start month and a month aggregation. 

#%%
print("1. Principal Component Forecast")

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# If variables are introduced from the command line, read them
if len(sys.argv) > 2:
    aggr = str(sys.argv[1])
    startmonth = int(sys.argv[2])
# If no variables were introduced, ask for them
else:
    # Monthly aggregation used
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m ]: ")

    # Which start month
    startmonth = int(input("Mes de inicialización (en número): "))


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
    # Array with month names
    endmonth_name = np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    # Dictionary to link the name of each valid month name to the endmonth and forecastmonth (in number)
    # Example: {validmonth_name1: [endmonth1, forecastMonth1], validmonth_name2: [endmonth2, forecastMonth2], validmonth_name3: [endmonth3, forecastMonth3]}
    initialization = {endmonth_name[(startmonth+(l+1) if startmonth+(l+1)<12 else startmonth+(l+1)-12)-1]: 
                      [startmonth+(l+1) if startmonth+(l+1)<=12 else startmonth+(l+1)-12, l+2] for l in reversed(range(lts))}
elif aggr=='3m': 
    # Array with 3-month season names
    endmonth_name = np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND'])
    # Dictionary to link the name of each valid season name to the startmonth and forecastmonth (in number)
    # Example: {validseason_name1: [endmonth1, forecastMonth1], validseason_name2: [endmonth2, forecastMonth2], validseason_name3: [endmonth3, forecastMonth3]}
    initialization = {endmonth_name[(startmonth+(l+3) if startmonth+(l+3)<12 else startmonth+(l+3)-12)-1]: 
                      [startmonth+(l+3) if startmonth+(l+3)<=12 else startmonth+(l+3)-12, l+4] for l in reversed(range(lts))}

# Dictionary with some other information
config = dict(
    hcstarty = 1993,
    hcendy = 2016,
    aggr = aggr,
)

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
MODESDATADIR = os.getenv('MODES_DIR') 
MODESDIR = MODESDATADIR + '/modes' # Directory where hindcast variability patterns files are located
CSVDIR = os.getenv('CSV_DIR') # Directory where variability patterns verification info (RPSS) is located

l_hcpcs_eofs_leads_val=list()
# For each eof (eof1, eof2, eof3, eof4) 
for m in range(4):
    l_hcpcs_eofx_leads_val=list()
    # For each leadtime (lead1, lead2, lead3)
    for lead in range(1,4):
        # Read skill scores for this eof
        eofx_rpss = pd.read_csv(CSVDIR+'/Score-card_rpss_'+aggr+'_eof'+str(m+1)+'.csv')
        # Rename model and lead columns
        eofx_rpss = eofx_rpss.rename(columns={eofx_rpss.columns[0]: "Model", eofx_rpss.columns[1]: "lead"})
        # Select forecastMonth and validmonth from initialization list
        validmonth = list(initialization)[-lead]
        fcmonth = initialization[validmonth][1]
        # Select desired month from scorecard table
        eofx_rpss_leadx = eofx_rpss[["Model","lead",validmonth]]
        # Sort models by forecast skill
        eofx_rpss_leadx = eofx_rpss_leadx[eofx_rpss_leadx['lead']=='lead'+str(lead)].sort_values(by=list(initialization)[-lead], ascending=False)
        # Select best model for this eof and leadtime
        origin = full_name[eofx_rpss_leadx['Model'].values[0]][0]
        system = full_name[eofx_rpss_leadx['Model'].values[0]][1]
        # Reading HCST data from file
        hcpcs_fname = f'{MODESDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
        # Read hindcast PCS data and select the desired eof and forecastMonth
        hcpcs_eofx_leadx = xr.open_dataset(hcpcs_fname).sel(mode=m,forecastMonth=fcmonth)
        # Quantify tercile thresholds
        low = hcpcs_eofx_leadx.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
        high = hcpcs_eofx_leadx.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
        # Quantify tercile averages
        low_value = hcpcs_eofx_leadx.where(hcpcs_eofx_leadx<low).mean(dim=['number','start_date']).pseudo_pcs
        med_value = hcpcs_eofx_leadx.where((hcpcs_eofx_leadx<=high) & (hcpcs_eofx_leadx>=low)).mean(dim=['number','start_date']).pseudo_pcs
        high_value = hcpcs_eofx_leadx.where(hcpcs_eofx_leadx>high).mean(dim=['number','start_date']).pseudo_pcs
        # Calculate the probability of each tercile (number of ensemble members above the threshold divided by the ensemble size), and assigning a representative value of each tercile
        hcpcs_eofx_leadx_low = (hcpcs_eofx_leadx.where(hcpcs_eofx_leadx<low).count(dim='number')/float(hcpcs_eofx_leadx.number.size)).assign_coords({'category':float(low_value)})
        hcpcs_eofx_leadx_med = (hcpcs_eofx_leadx.where((hcpcs_eofx_leadx<=high) & (hcpcs_eofx_leadx>=low)).count(dim='number')/float(hcpcs_eofx_leadx.number.size)).assign_coords({'category':float(med_value)})
        hcpcs_eofx_leadx_high = (hcpcs_eofx_leadx.where(hcpcs_eofx_leadx>high).count(dim='number')/float(hcpcs_eofx_leadx.number.size)).assign_coords({'category':float(high_value)})
        # Concatenate the three terciles
        hcpcs_eofx_leadx_ter = xr.concat([hcpcs_eofx_leadx_low,hcpcs_eofx_leadx_med,hcpcs_eofx_leadx_high],dim='category')
        # Find the most probable tercile and assign the corresponding value
        hcpcs_eofx_leadx_val = hcpcs_eofx_leadx_ter.idxmax(dim='category').drop('mode')
        # Include also information about the variability patterns forecast skill (RPSS)
        hcpcs_eofx_leadx_val = hcpcs_eofx_leadx_val.assign_coords({'rpss':eofx_rpss_leadx[list(initialization)[-lead]].values[0]})
        # Append results for each leadtime
        l_hcpcs_eofx_leads_val.append(hcpcs_eofx_leadx_val.assign_coords({'forecastMonth':fcmonth}))
    # Concatenate results for all leadtimes         
    hcpcs_eofx_leads_val = xr.concat(l_hcpcs_eofx_leads_val,dim='forecastMonth')        
    # Append results for each eof          
    l_hcpcs_eofs_leads_val.append(hcpcs_eofx_leads_val.assign_coords({'mode':m}))
# Concatenate results for all eofs                 
hcpcs_eofs_leads_val = xr.concat(l_hcpcs_eofs_leads_val,dim='mode')                    

# Directory selection
POSTDIR = os.getenv('POST_DIR')
NEWMODESDIR = POSTDIR + '/modes' # Directory where variability patterns forecasts will be saved
# Check if the directory exists
if not os.path.exists(NEWMODESDIR):
    # If it doesn't exist, create it
    try:
        os.makedirs(NEWMODESDIR)
    except FileExistsError:
        pass

# Save pcs best forecasts
hcpcs_eofs_leads_val.to_netcdf(f'{NEWMODESDIR}/PCs_best_forecasts_stmonth{startmonth:02d}.'+aggr+'.nc')
