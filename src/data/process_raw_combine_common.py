import os.path
if __name__ == '__main__': print('running process_raw_combine_common.py')

import pandas as pd

# TODO: Set relative path if needed (doesnt work right now but maybe for server)
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = 'C:\\Users\\langh\\Individual_project\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

#### Combine variables and flatten the structure ####
df_spei = pd.read_csv(interrimpath + 'spei_processed.csv', index_col=0)
df_cruts = pd.read_csv(interrimpath + 'cruts_processed.csv', index_col=0)

df_spei.set_index(['lat', 'lon', 'month', 'year'], inplace=True)
df_cruts.set_index(['lat', 'lon', 'month', 'year'], inplace=True)

df_var = df_spei.join(df_cruts, how='outer')
df_var.reset_index()

df_var.to_csv(interrimpath + 'df_var_common_mth.csv')
del df_var
