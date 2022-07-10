
import pandas as pd
import numpy as np
import random
import sys

# sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append("..")
try:
    import utils as ut
    print('utils loaded from ..')
except:
    print('no module in ..')
    try:
        import src.utils as ut
        print('utils loaded from src')
    except:
        print('no module src')


# Parameters
harvest_month = 5  # mth used for SPEI and
pct_train = 0.90  # training/test splits
random.seed(10)

## combining sets
datapath = "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\"
interrimpath = datapath + "interrim/"
processedpath = datapath + "processed/"

# indexes
crops = ["wheat_winter", "wheat_spring", "maize", "rice", "soy"]
coords = ['lon', 'lat', 'year']

# Load feature subsets with positive yields
df_spei = pd.read_csv(interrimpath + "df_spei_mth_8017.csv", index_col=0)
df_cruts_mth = pd.read_csv(interrimpath + 'df_cruts_mth.csv', index_col=0)
df_heat = pd.read_csv(interrimpath + "df_heat_mth.csv", index_col=0)

# Convert CRUTS from Monthly to Year structure
df_cruts = ut.df_mth_to_year(df_cruts_mth)

# subset SPEI only for harvest month
df_spei = df_spei[df_spei['month'] == harvest_month]

# combine SPEI and CRUTS as these do not vary per crop
df_comb = df_spei.merge(df_cruts, on=['lon', 'lat', 'year'], how='inner')
df_comb.drop('month', axis = 1, inplace = True)

# Subset SPEI only for harvest month
df_heat = df_heat[df_heat['month'] == harvest_month]
df_heat.drop('month', axis = 1, inplace = True)

## Make datasets for each crop
for crop in crops:
    print(crop)
    croppath = processedpath + crop + '/'

    # Subset heat to relevant crop
    if crop == 'maize' or crop == 'rice':
        c = 'maizerice'
    elif crop == 'wheat_spring' or crop == 'wheat_winter':
        c = 'wheat'

    df_heat_c = df_heat[coords + [c + '_mth', c + '_6mths']]

    # Make combined dataset with features
    df_comb_crop = ut.fast_join(df_comb, df_heat_c, ['lon', 'lat', 'year'], 'inner')
    df_comb = df_comb.reset_index()

    # Make full dataset with yields
    df_yield = pd.read_csv(interrimpath + 'yield\\df_' + crop + '_yield.csv', index_col=0)
    df_comb_crop = ut.fast_join(df_comb_crop, df_yield, ['lon', 'lat', 'year'], 'inner')
    df_comb_crop = df_comb_crop.reset_index()

    # Output training and test set w pct_train % split randomly
    num_rows = len(df_comb_crop)
    num_train = round(num_rows * pct_train)
    idx = np.arange(num_rows)
    np.random.shuffle(idx)
    train_idx = idx[0:num_train]
    test_idx = idx[num_train:]

    df_comb_crop.iloc[train_idx].to_csv(croppath + 'df_' + crop + "_historical_mix_train90.csv")
    df_comb_crop.iloc[test_idx].to_csv(croppath + 'df_' + crop + "_historical_mix_test10.csv")

    # Output training and test set with pct_train unique pixels held out for test set
    df_coords = df_comb_crop.groupby(['lon', 'lat']).size().reset_index()

    num_rows = len(df_coords)
    num_train = round(num_rows * pct_train)
    idx = np.arange(num_rows)
    np.random.shuffle(idx)
    train_idx = idx[0:num_train]
    test_idx = idx[num_train:]

    df_coords = df_coords[['lon', 'lat']]
    df_coords_train = df_coords.iloc[train_idx]
    df_coords_test = df_coords.iloc[test_idx]

    df_comb_train = ut.fast_join(df_comb_crop, df_coords_train, ['lon', 'lat'], 'inner')
    df_comb_crop = df_comb_crop.reset_index()

    df_comb_test = ut.fast_join(df_comb_crop, df_coords_test, ['lon', 'lat'], 'inner')
    df_comb_crop = df_comb_crop.reset_index()

    df_comb_train.to_csv(croppath + 'df_' + crop + "_historical_pix_train90.csv")
    df_comb_test.to_csv(croppath + 'df_' + crop + "_historical_pix_test10.csv")

    # Output training and test set last year held out
    years_hold_out = 1

    df_comb_pre = df_comb_crop[df_comb_crop['year'] <= (max(df_comb_crop['year']) - years_hold_out)]
    df_comb_post = df_comb_crop[df_comb_crop['year'] > (max(df_comb_crop['year']) - years_hold_out)]

    df_comb_pre.to_csv(croppath + 'df_' + crop + "_historical_1yearshold_pre.csv")
    df_comb_post.to_csv(croppath + 'df_' + crop + "_historical_1yearshold_post.csv")

    # Output training and test set w years_hold_out in test set
    years_hold_out = 5

    df_comb_pre = df_comb_crop[df_comb_crop['year'] <= (max(df_comb_crop['year']) - years_hold_out)]
    df_comb_post = df_comb_crop[df_comb_crop['year'] > (max(df_comb_crop['year']) - years_hold_out)]

    df_comb_pre.to_csv(croppath + 'df_' + crop + "_historical_5yearshold_pre.csv")
    df_comb_post.to_csv(croppath + 'df_' + crop + "_historical_5yearshold_post.csv")

