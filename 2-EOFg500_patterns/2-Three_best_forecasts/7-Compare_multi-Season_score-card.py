# %% [markdown]
# # 7. Compare multi-season score cards

# This script is used to compare score-cards, 
# which show the difference between two sets of verification score values for forecasts of all months or seasons with different systems and initializations. 
#
# First we have to decide a verification score. 

#%%
print("7. Compare multi-season score cards")

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import numpy as np
import pandas as pd
import locale
import calendar
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import warnings
warnings.filterwarnings('ignore')

# If variables are introduced from the command line, read them
if len(sys.argv) > 2:
    var = str(sys.argv[1])
    aggr = str(sys.argv[2])
    score = str(sys.argv[3])
# If no variables were introduced, ask for them
else:
    # Subset of plots to be produced
    var = input("Selecciona la variable [ t2m , tprate , msl ]: ")

    # Subset of plots to be produced
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m ]: ")

    # Verification score
    score = input("Usar el siguiente score [ bs , corr , roc , rocss , rps , rpss ]: ")

# Dictionary to link full system names and simplier names
full_name = {#'ECMWF-System 4': ['ecmwf','4'],
             #'ECMWF-SEAS5': ['ecmwf', '5'],
             'ECMWF-SEAS5.1': ['ecmwf', '51'],
             #'Météo France-System 5': [ 'meteo_france', '5'],
             #'Météo France-System 6': [ 'meteo_france', '6'],
             #'Météo France-System 7': [ 'meteo_france', '7'],
             'Météo France-System 8': [ 'meteo_france', '8'],
             #'Met Office-System 12': ['ukmo', '12'],
             #'Met Office-System 13': ['ukmo', '13'],
             #'Met Office-System 14': ['ukmo', '14'],
             #'Met Office-System 15': ['ukmo', '15'],
             #'Met Office-GloSea6': ['ukmo', '600'],
             #'Met Office-GloSea6.1': ['ukmo', '601'],
             'Met Office-GloSea6.2': ['ukmo', '602'],
             #'Met Office-GloSea6.3': ['ukmo', '603'],
             #'DWD-GCFS2.0': ['dwd', '2'],
             'DWD-GCFS2.1': ['dwd', '21'],
             #'CMCC-SPSv3.0': ['cmcc', '3'],
             'CMCC-SPSv3.5': ['cmcc', '35'],
             'NCEP-CFSv2': ['ncep', '2'],
             #'JMA-CPS2': ['jma', '2'],
             'JMA-CPS3': ['jma', '3'],
             #'ECCC-GEM-NEMO': ['eccc', '1'],
             #'ECCC-CanCM4i': ['eccc', '2'],
             'ECCC-GEM5-NEMO': ['eccc', '3']
            }

# Number of lead-times
lts=3

load_dotenv() # Data is saved in a path defined in file .env
CSVDIR = os.getenv('DEFAULT_VER')

# Common labels to be used in plot titles
VARNAMES = {
    't2m' : '2-metre temperature',
    'tprate'  : 'total precipitation',    
    'msl' : 'mean-sea-level pressure',
}

if aggr=='1m':
    var_seasons = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
elif aggr=='3m': 
    var_seasons = ["JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF"]

# Some predefined options to plot each score
score_options = {'bs': [np.linspace(-0.3,0.3,10), plt.colormaps['bwr_r'], 3, 'max', 'Brier Score (BS)'],
                 'corr': [np.linspace(-0.5,0.5,10), plt.colormaps['bwr_r'], 1, 'both', 'Spearmans Rank Correlation'],
                 'roc': [np.linspace(-0.3,0.3,10), plt.colormaps['bwr_r'], 3, 'both', 'Area under Relative Operating Characteristic (ROC) curve'],
                 'rocss': [np.linspace(-0.3,0.3,10), plt.colormaps['bwr_r'], 3, 'both', 'Relative Operating Characteristic Skill Score (ROCSS)'],
                 'rps': [np.linspace(-0.3,0.3,10), plt.colormaps['bwr_r'], 1, 'max', 'Ranked Probability Score (RPS)'],
                 'rpss': [np.linspace(-0.3,0.3,10), plt.colormaps['bwr_r'], 1, 'both', 'Ranked Probability Skill Score (RPSS)'],
                }

new_csv = f'./plots/scorecards/Score-card_{score}_{aggr}_{var}.csv'
old_csv = f'{CSVDIR}/Score-card_{score}_{aggr}_{var}.csv'

if score_options[score][2]>1:
    df_new = pd.read_csv(new_csv, header=[0,1])
    df_old = pd.read_csv(old_csv, header=[0,1])

else:
    df_new = pd.read_csv(new_csv)
    df_old = pd.read_csv(old_csv)

DATA = df_new.iloc[:,2:] - df_old.iloc[:,2:]

# Size of solution's array
n_models = len(full_name)
n_init = lts
n_seasons = len(var_seasons)
n_terciles = score_options[score][2]

# %% [markdown]
# ## 7.1 Score-cards

# Then we represent the results.

#%%
print("7.1 Score-cards")

# Directory creation
PLOTSDIR = f'./plots/scorecards'
# Check if the directory exists
if not os.path.exists(PLOTSDIR):
    # If it doesn't exist, create it
    try:
        os.makedirs(PLOTSDIR)
    except FileExistsError:
        pass

# Prepare strings for titles
locale.setlocale(locale.LC_ALL, 'en_GB')
# if aggr=='1m':
#     validmonth = config['start_month'] + (fcmonth-1)
#     validmonth = validmonth if validmonth<=12 else validmonth-12
#     tit_line2 = f'Valid month: {calendar.month_abbr[validmonth].upper()}'
# elif aggr=='3m':
#     validmonths = [vm if vm<=12 else vm-12 for vm in [config['start_month'] + (fcmonth-1) - shift for shift in range(3)]]
#     validmonths = [calendar.month_abbr[vm][0] for vm in reversed(validmonths)]
#     tit_line2 = f'Valid months: {"".join(validmonths)}'
# else:
#     raise BaseException(f'Unexpected aggregation {aggr}')
tit_line1 = f'{VARNAMES[var]}'+f'\n Difference in {score_options[score][4]}'
figname = f'{PLOTSDIR}/Dif_score-card_{score}_{aggr}_{var}.png'

# Create figure
fig = plt.figure(figsize=(17,8))
ax = fig.add_subplot()

# Representation
x = np.arange(DATA.shape[1])
y = np.arange(DATA.shape[0])
levels = score_options[score][0]
cmap = score_options[score][1]
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
c = ax.pcolor(DATA, edgecolors='w', linewidths=4, cmap=cmap, norm=norm)
cb = plt.colorbar(c,
                orientation='horizontal',
                location='bottom',
                aspect=60,
                pad=0.05,
                label=score.upper())

# Labels and title
ax.set_yticks(y[::n_init]+0.5*n_init,[label for label in full_name])
ax.set_xticks(x[::n_terciles]+0.5*n_terciles,[tick for tick in var_seasons])
init_labels = ['lead3', 'lead2', 'lead1'] *n_models
ax.set_yticks(y+0.499,init_labels,minor=True)
if n_terciles>1:
    terciles_labels = ['lower', 'middle', 'upper'] * n_seasons
    ax.set_xticks(x+0.499,terciles_labels,minor=True)
ax.tick_params(axis='x', which="minor", 
               bottom=False, left=False, top=True, right=True,
               labelbottom=False, labelleft=False, labeltop=True, labelright=True,
               labelrotation=25.)
ax.tick_params(axis='y', which="minor", 
               bottom=False, left=False, top=True, right=True,
               labelbottom=False, labelleft=False, labeltop=True, labelright=True,
               labelrotation=0.)
plt.hlines(y[::n_init],0,len(x),color='k')
plt.vlines(x[::n_terciles],0,len(y),color='k')
plt.title(tit_line1, fontsize=14, loc='center')

# Numbers inside box
for y in range(DATA.shape[0]):
    for x in range(DATA.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.2f' % DATA.values[y,x],
                    horizontalalignment='center',
                    verticalalignment='center',
                )

# Save figure
plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.05)
fig.savefig(figname,dpi=600,bbox_inches='tight')  

