#%% [markdown]

# # 4. Multi-system verification plots

# This script is used to create verification score maps for surface variables for all forecasting system in the same figure. 
# 
# First we have to decide a start month a month aggregation and a end month. 

#%%
print("4. Multi-system verification plots")  

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import numpy as np
import locale
import calendar
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

# If variables are introduced from the command line, read them
if len(sys.argv) > 2:
    startmonth = int(sys.argv[1])
    aggr = str(sys.argv[2])
    fcmonth = int(sys.argv[3])
    score = str(sys.argv[4])
    region = str(sys.argv[5])
# If no variables were introduced, ask for them
else:
    # Which start month
    startmonth = int(input("Mes de inicialización (en número): "))

    # Subset of plots to be produced
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m ]: ")

    # Forecast month
    if aggr=='1m':
        answ = input("Resultados para el mes [ Jan , Feb , Mar , Apr , May , Jun , Jul , Aug , Sep , Oct , Nov , Dec ]: ")
        endmonth = np.where(np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']) == answ)[0][0]+1
        fcmonth = (endmonth-startmonth)+1 if (endmonth-startmonth)>=0 else (endmonth-startmonth)+13
    else:
        answ = input("Resultados para el trimestre ['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']: ")
        endmonth = np.where(np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']) == answ)[0][0]+1
        fcmonth = (endmonth-startmonth)+1 if (endmonth-startmonth)>=0 else (endmonth-startmonth)+13

    # Verification score
    score = input("Usar el siguiente score [ bs , corr , roc , rocss , rps , rpss ]: ")

    # Region selected for plots
    region = input("Selecciona qué región representar [ Iberia , MedCOF ]: ")

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

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
POSTDIR = os.getenv('POST_DIR')
NEWSCOREDIR = POSTDIR + '/scores'
NEWPLOTSDIR = f'./plots'

# Common labels to be used in plot titles
VARNAMES = {
    't2m' : '2-metre temperature',
    'msl' : 'mean-sea-level pressure',
    'tprate'  : 'total precipitation'
}
CATNAMES=['lower tercile', 'middle tercile', 'upper tercile']

# Some predefined options to plot each score
score_options = {'bs': [np.linspace(0.,0.5,11), plt.colormaps['YlGn'], 3, 'max', 'Brier Score (BS)'],
                 'corr': [np.linspace(-1.,1.,11), plt.colormaps['RdYlBu_r'], 1, 'both', 'Spearmans Rank Correlation (stippling where significance below 95%)'],
                 'roc': [np.linspace(0.,1.,11), plt.colormaps['BrBG'], 3, 'both', 'Area under Relative Operating Characteristic (ROC) curve'],
                 'rocss': [np.linspace(-0.5,0.5,9), plt.colormaps['BrBG'], 3, 'both', 'Relative Operating Characteristic Skill Score (ROCSS)'],
                 'rps': [np.linspace(0.3,0.5,11), plt.colormaps['YlGn_r'], 1, 'max', 'Ranked Probability Score (RPS)'],
                 'rpss': [np.linspace(-0.5,0.5,11), plt.colormaps['BrBG'], 1, 'both', 'Ranked Probability Skill Score (RPSS)'],
                }

# Region definition
if region=='Iberia':
    box_limits = [-30, 5, 25, 50] # [West, East, South, North]
elif region=='MedCOF':
    box_limits = [-30, 50, 14, 55] # [West, East, South, North]

# Change titles font size
plt.rc('axes', titlesize=20)

# For each variable
for var in VARNAMES:
    # For each quantile
    for icat in range(score_options[score][2]):
        # Create figure
        fig = plt.figure(figsize=(12,18))
        # Subdivide figure in a 4x2 grid
        gs = fig.add_gridspec(4,2)
        m = 0
        # For each model
        for label in full_name:

            # Save the simplier model and system name
            model = full_name[label][0]
            system = full_name[label][1]
            # Here we save the configuration
            config = dict(
                hcstarty = 1993,
                hcendy = 2016,
                start_month = startmonth,
                origin = model,
                system = system,
            )
            # Base name for hindcast
            hcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)

            # Prepare strings for titles
            locale.setlocale(locale.LC_ALL, 'en_GB')
            tit_line1 = f'{label}'
            tit_line2_base = f'Start month: {calendar.month_abbr[config["start_month"]].upper()}'
            if aggr=='1m':
                validmonth = config['start_month'] + (fcmonth-1)
                validmonth = validmonth if validmonth<=12 else validmonth-12
                tit_line2 = tit_line2_base + f' - Valid month: {calendar.month_abbr[validmonth].upper()}'
            elif aggr=='3m':
                validmonths = [vm if vm<=12 else vm-12 for vm in [config['start_month'] + (fcmonth-1) - shift for shift in range(3)]]
                validmonths = [calendar.month_abbr[vm][0] for vm in reversed(validmonths)]
                tit_line2 = tit_line2_base + f' - Valid months: {"".join(validmonths)}'
            else:
                raise BaseException(f'Unexpected aggregation {aggr}')
            
            # Check if file exists
            if not os.path.exists(f'{NEWSCOREDIR}/{hcst_bname}.{aggr}.{score}.nc'):
                print('No se creó el archivo con dicho score: ')
                print(f'{NEWSCOREDIR}/{hcst_bname}.{aggr}.{score}.nc')
                sys.exit()

            # Read the data file
            sc = xr.open_dataset(f'{NEWSCOREDIR}/{hcst_bname}.{aggr}.{score}.nc')
  
            # Select forecast month
            thissc = sc.sel(forecastMonth=fcmonth)

            # Create subplot (This indicates grid position of subplot: m-4*(m//4),m//4)
            ax = fig.add_subplot(gs[m-4*(m//4),m//4],projection=ccrs.PlateCarree())
            ax.set_extent(box_limits, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
            gl = ax.gridlines(draw_labels=True)
            gl.ylabels_right = False

            # Select quantile (if score is splitted into quantiles)
            if score_options[score][2]>1:
                avalues = thissc.sel(category=icat)[var].values
            else:
                avalues = thissc[var].values
  
            # Check if data values matrices need to be transposed
            if avalues.T.shape == (thissc[var].lat.size, thissc[var].lon.size):
                avalues = avalues.T
            elif avalues.shape == (thissc[var].lat.size, thissc[var].lon.size):
                pass                           
            else:
                raise BaseException(f'Unexpected data value matrix shape: {avalues.shape}' )

            # Represent
            cs = plt.contourf(thissc[var].lon,thissc[var].lat,avalues,levels=score_options[score][0],cmap=score_options[score][1],extend=score_options[score][3])
            # If score is corr, we also represent p-values
            if score=='corr':
                corr_pval = xr.open_dataset(f'{NEWSCOREDIR}/{hcst_bname}.{aggr}.{score}_pval.nc')
                thiscorrpval = corr_pval.sel(forecastMonth=fcmonth)
                corrpvalvalues = thiscorrpval[var].values
                if corrpvalvalues.T.shape == (thiscorrpval[var].lat.size, thiscorrpval[var].lon.size):
                    corrpvalvalues = corrpvalvalues.T
                elif corrpvalvalues.shape == (thiscorrpval[var].lat.size, thiscorrpval[var].lon.size):
                    pass                           
                else:
                    raise BaseException(f'Unexpected data value matrix shape: {corrpvalvalues.shape}' )
                plt.contourf(thiscorrpval[var].lon,thiscorrpval[var].lat,corrpvalvalues,levels=[0.05,np.inf],hatches=['...',None],colors='none')
            plt.title(tit_line1,loc='center',fontsize=12)      
            m+=1
        # Add colorbar
        cbar_ax = fig.add_axes([0.05, 0.05, 0.9, 0.01])
        cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal',label=score.upper())
        # Add title
        if score_options[score][2]==3:
            fig.suptitle(f'{score_options[score][4]} \n{VARNAMES[var]}' + f' ({CATNAMES[icat]})\n' + tit_line2, fontsize=16)
            figname = f'{NEWPLOTSDIR}/stmonth{config["start_month"]:02d}/All_models_stmonth{config["start_month"]:02d}_hindcast{config["hcstarty"]}-{config["hcendy"]}_monthly.{region}.{aggr}.{"".join(validmonths)}.{var}.{score}{str(icat)}.png'
        else:
            fig.suptitle(f'{score_options[score][4]} \n{VARNAMES[var]}\n' + tit_line2, fontsize=16)
            figname = f'{NEWPLOTSDIR}/stmonth{config["start_month"]:02d}/All_models_stmonth{config["start_month"]:02d}_hindcast{config["hcstarty"]}-{config["hcendy"]}_monthly.{region}.{aggr}.{"".join(validmonths)}.{var}.{score}.png'
        # Controlling subplot distances
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.07)
        # Save figure
        fig.savefig(figname)  
