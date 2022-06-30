import pandas as pd
from src.utils import maximumZero

## combining sets
datapath = sys.path[len(sys.path) - 1] + "/data/"
interrimpath = datapath + "interrim/"

crops = ["wheat", "maize", "rice", "soy"]

crops = ['maize', 'soybean', 'wheat', 'rice']
df_cruts = pd.read_csv(interrimpath + "cruts_processed.csv", index_col=0)
df_spei = pd.read_csv(interrimpath + "spei_processed.csv", index_col=0)
df_heat = pd.read_csv(interrimpath + "cpc_tmax_processed.csv",  index_col=0)

# Calculate extreme heat features
df_heat = df_heat.dropna()
df_heat = df_heat.sort_values(by=['lon', 'lat', 'year', 'month'])

df_heat.drop('tmax', axis=1, inplace=True)

for crop in crops:
    df_heat[crop + '_extheat_6mth_'] = df_heat[crop + 'max'].rolling(6).sum().apply(maximumZero)
    df_heat.drop(crop + 'max', axis=1, inplace=True)

# Change to horisontal structure
df_heat.drop('month', axis=1, inplace=True)
df_heat = (df_heat.set_index(['lon', 'lat', 'year',
                     df_heat.groupby(['lon', 'lat', 'year']).cumcount().add(1)])
       .unstack()
       .sort_index(axis=1, level=1))
df_heat.columns = [f'{a}{b}' for a, b in df_heat.columns]
df_heat = df_heat.reset_index()

# Change CRUTS from vertical to horisontal set with yearly index
df_cruts = df_cruts.sort_values(by=['lon', 'lat', 'year', 'month'])
df_cruts.drop('month', axis=1, inplace=True)
df_cruts = (df_cruts.set_index(['lon', 'lat', 'year',
                     df_cruts.groupby(['lon', 'lat', 'year']).cumcount().add(1)])
       .unstack()
       .sort_index(axis=1, level=1))
df_cruts.columns = [f'{a}{b}' for a, b in df_cruts.columns]
df_cruts = df_cruts.reset_index()

## Make datasets for each crop
for crop in crops:
    # Use rolling sum of events only for may
    df_heat = df_heat[['lon', 'lat', crop + 'ext_heat_6mth', 'month', 'year']]
    df_heat = df_heat[df_heat['month'] == harvest_month]
    df_spei = df_spei[df_spei['month'] == harvest_month]

    # Combine crop, heat and SPEI datasets per year
    df_comb = pd.read_csv(datapath + crop + "_combined.csv")
    df_comb = df_comb.merge(df_spei, on=['lon', 'lat', 'year'], how='inner')
    df_comb = df_comb.merge(df_heat, on=['lon', 'lat', 'year'], how='inner')
    df_comb = df_cruts.merge(df_comb, on=['lon', 'lat', 'year'], how='left').dropna()

    #   df_comb = df_comb[['yield', 'lon', 'lat', 'year', 'SPEI', 'extHeat', 'tmp', 'cld', 'dtr',
    #                  'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']]

    df_comb.to_csv(datapath + "historical_combined.csv")
    df_comb = pd.read_csv(datapath + "historical_combined.csv", index_col=0)

    # Output training and test set w pct_train % split randomly
    pct_train = 0.90
    num_rows = len(df_comb)
    num_train = round(num_rows * pct_train)
    idx = np.arange(num_rows)
    np.random.shuffle(idx)
    train_idx = idx[0:num_train]
    val_idx = idx[num_train:]

    df_comb.iloc[train_idx].to_csv(datapath + "historical_combined_train90.csv")
    df_comb.iloc[val_idx].to_csv(datapath + "historical_combined_test10.csv")

    # Output training and test set with pct_train unique pixels held out for test set
    # TODO

    # Output training and test set last year held out
    # TODO

    # Output training and test set w years_hold_out in test set
    years_hold_out = 5

    df_comb_pre = df_comb[df_comb['year'] < (max(df_comb['year']) - 5)]
    df_comb_post = df_comb[df_comb['year'] >= (max(df_comb['year']) - 5)]

    df_comb_pre.to_csv(datapath + "historical_combined_5yearshold_pre.csv")
    df_comb_post.to_csv(datapath + "historical_combined_5yearshold_post.csv")
