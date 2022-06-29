## combining sets
datapath = sys.path[len(sys.path)-1] + "/data/"
interrimpath = datapath + "interrim/"

df_wheat = pd.read_csv(datapath + "wheat_combined.csv")
df_cruts = pd.read_csv(datapath + "cruts_processed.csv")
df_spei = pd.read_csv(datapath + "spei_processed.csv")
df_heat = pd.read_csv(datapath + "cpc_tmax.csv")

# Calculate extreme heat features

df_heat = df_heat.dropna()

df_tmax_out = df_tmax_out.sort_values(by=['lon', 'lat', 'year', 'month'])

for crop in ["wheat", "maize", "rice", "soy"]:
    df_tmax_out[crop + '6mthRollMax'] = df_tmax_out[crop + 'max'].rolling(6).sum()


# Use rolling sum of events only for may
df_heat = df_heat[['lon', 'lat', 'wheat6mthRollMax', 'month', 'year']]
df_heat = df_heat[df_heat['month'] == 5]
df_spei = df_spei[df_spei['month'] == 5]

df_comb = df_wheat.merge(df_spei, on=['lon', 'lat', 'year'], how='inner')
df_comb = df_comb.merge(df_heat, on=['lon', 'lat', 'year'], how='inner')
df_comb = df_cruts.merge(df_comb, on=['lon', 'lat', 'year'], how='left').dropna()

df_comb.rename(columns={'rolling': 'SPEI', 'wheat6mthRollMax': 'extHeat'}, inplace=True)
df_comb = df_comb[['yield', 'lon', 'lat', 'year', 'SPEI', 'extHeat', 'tmp', 'cld', 'dtr',
                   'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']]

df_comb['wet'] = df_comb['wet'].apply(timeToDays)
df_comb['frs'] = df_comb['frs'].apply(timeToDays)

df_comb.to_csv(datapath + "historical_combined.csv")
df_comb = pd.read_csv(datapath + "historical_combined.csv",  index_col=0)


# Set up training and test set with random samples
pct_train = 0.90
num_rows = len(df_comb)
num_train = round(num_rows * pct_train)
idx = np.arange(num_rows)
np.random.shuffle(idx)
train_idx = idx[0:num_train]
val_idx = idx[num_train:]

df_comb.iloc[train_idx].to_csv(datapath + "historical_combined_train90.csv")
df_comb.iloc[val_idx].to_csv(datapath + "historical_combined_test10.csv")

# Set up training and test set hold out of last years
years_hold_out = 5

df_comb_pre = df_comb[df_comb['year'] < (max(df_comb['year']) - 5)]
df_comb_post = df_comb[df_comb['year'] >= (max(df_comb['year']) - 5)]

df_comb_pre.to_csv(datapath + "historical_combined_5yearshold_pre.csv")
df_comb_post.to_csv(datapath + "historical_combined_5yearshold_post.csv")
