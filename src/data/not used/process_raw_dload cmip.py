
import sys
import os
import requests
import sys
from tqdm import tqdm

sys.path.append("..")
sys.path.append(".")

try:
    import utils as ut
    import maps as mp
    print('utils loaded from ..')
except:
    print('no module in ..')
    try:
        import src.utils as ut
        import src.h_geo as mp
        print('utils loaded from src')
    except:
        print('no module src')

print('Packages loaded!')

current_directory = os.getcwd()
datapath = current_directory + "/data/"
rawpath = datapath + "raw/"
cmip_raw = rawpath + "clim_mod/"
interrimpath = datapath + "interrim/"
cmip_int = interrimpath + 'cmip/'

# CRUTS vars:
# vars = ['cld', 'dtr', 'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']

cruts_to_cmip = {
    'cld': 'clt',      # cloud cover
    'tmp': 'tas',      # mean temperature
    'tmn': 'tasmin',   # min temperatu
    'tmx': 'tasmax',   # max temp
    'pre': 'pr',         # precipitation
}

vars_cruts = ['tmp', 'tmn', 'tmx', 'pre', 'cld']
vars_cmip = []
for v in vars_cruts:
    vars_cmip = vars_cmip + [cruts_to_cmip[v]]


def download_cmip(url, file):
    session = requests.Session()
    r = session.get(url, stream = True)
    r.raise_for_status()
    with open(file,"wb") as out:
        for chunk in tqdm(r.iter_content(chunk_size=1024 * 1024)):
            if chunk:
                out.write(chunk)

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

for scenario in tqdm(scenarios):
    for var in tqdm(vars_cmip):
        print(var, scenario)
        can_esm_url = 'http://crd-esgf-drc.ec.gc.ca/thredds/fileServer/esgD_dataroot/AR6/CMIP6/ScenarioMIP/CCCma/CanESM5/' + scenario + '/r1i1p1f1/Amon/'+ var + '/gn/v20190429/'+ var +'_Amon_CanESM5_' + scenario +'_r1i1p1f1_gn_201501-210012.nc'
        access_esm_url = 'http://esgf.nci.org.au/thredds/fileServer/master/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/' + scenario +'/r1i1p1f1/Amon/'+ var + '/gn/v20191115/'+ var + '_Amon_ACCESS-ESM1-5_'+ scenario + '_r1i1p1f1_gn_201501-210012.nc'
        miroc_url = 'http://esgf-data2.diasjp.net/thredds/fileServer/esg_dataroot/CMIP6/ScenarioMIP/MIROC/MIROC6/ssp585/r1i1p1f1/Amon/'+ var + '/gn/v20190627/'+ var + '_Amon_MIROC6_'+ scenario + '_r1i1p1f1_gn_201501-210012.nc'
        can_esm_file = cmip_raw + 'CanESM5/cruts/CanESM5_' + var +'_' + scenario + '.nc'
        access_esm_file = cmip_raw + 'ACCESS-ESM1-5/cruts/ACCESS-ESM1-5_' + var +'_' + scenario + '.nc'
        miroc_file = cmip_raw + 'MIROC6/cruts/MIROC6_' + var +'_' + scenario + '.nc'
        if not os.path.exists(can_esm_file):
            print('downloading CanESM', var, scenario)
            download_cmip(can_esm_url, can_esm_file)
        
        if not os.path.exists(access_esm_file):
            print('downloading AccesESM', var, scenario)
            download_cmip(access_esm_url, access_esm_file)
        
        if not os.path.exists(miroc_file):
            print('downloading MIROC', var, scenario)
            download_cmip(miroc_url, miroc_file)
        
        print(var, scenario, 'Done!')
print('Done!')