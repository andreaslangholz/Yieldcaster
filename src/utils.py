import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from numpy import sqrt, NaN
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
import folium
from shapely.geometry import Point
from geopandas import GeoDataFrame
import shapely
import geopandas as gpd
import re

print(__package__)
print(__name__)

class makeDataset(Dataset):
    def __init__(self, df, target, mode='train'):
        self.mode = mode
        self.df = df

        if self.mode == 'train':
            self.df = self.df.dropna()
            self.oup = self.df.pop(target).values.reshape(len(df), 1)
            self.inp = self.df.values
        else:
            self.inp = self.df.values

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt = torch.Tensor(self.inp[idx])
            oupt = torch.Tensor(self.oup[idx])
            return {'inp': inpt,
                    'oup': oupt
                    }
        else:
            inpt = torch.Tensor(self.inp[idx])
            return {'inp': inpt
                    }


def rmse(actual, pred):
    return sqrt(mean_squared_error(actual, pred))


def maximumZero(x):
    if x > 0:
        return x
    else:
        return 0

def recenter_lon(lon):
    if lon > 180:
        return lon - 360
    else:
        return lon


def df_mth_to_year(df):
    df_var_horizontal = df.sort_values(by=['lon', 'lat', 'year', 'month'])
    df_var_horizontal.drop('month', axis=1, inplace=True)

    try:
        df_var_horizontal.drop('time', axis=1, inplace=True)
    except:
        print('no time column')

    df_var_horizontal = (df_var_horizontal.set_index(['lon', 'lat', 'year',
                                                      df_var_horizontal.groupby(['lon', 'lat', 'year']).cumcount().add(
                                                          1)])
                         .unstack()
                         .sort_index(axis=1, level=1))
    df_var_horizontal.columns = [f'{a}{b}' for a, b in df_var_horizontal.columns]
    df_var_horizontal = df_var_horizontal.reset_index()
    return df_var_horizontal


def fast_join(df1, df2, list_to_join_on, how='outer'):
    df1.set_index(list_to_join_on, inplace=True)
    df2.set_index(list_to_join_on, inplace=True)
    df_out = df1.join(df2, how=how, on=list_to_join_on, lsuffix='left', rsuffix='right')
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df_out = df_out.reset_index()
    return df_out


def get_square_around_point(point_geom, delta_size=0.25):
    point_coords = np.array(point_geom.coords[0])

    c1 = point_coords + [-delta_size, -delta_size]
    c2 = point_coords + [-delta_size, +delta_size]
    c3 = point_coords + [+delta_size, +delta_size]
    c4 = point_coords + [+delta_size, -delta_size]

    square_geom = shapely.geometry.Polygon([c1, c2, c3, c4])

    return square_geom


def get_gdf_with_squares(gdf_with_points, delta_size=0.25):
    gdf_squares = gdf_with_points.copy()
    gdf_squares['geometry'] = (gdf_with_points['geometry']
                               .apply(get_square_around_point,
                                      delta_size))

    return gdf_squares


def sub_mask(df, mask='yield'):
    if mask == 'yield':
        df_mask = pd.read_csv("C:\\Users\\langh\\Individual_project\\Yieldcaster\\data\\interrim\\yield_mask.csv",
                              index_col=0)
    if mask == 'land':
        df_mask = pd.read_csv("C:\\Users\\langh\\Individual_project\\Yieldcaster\\data\\interrim\\landmask.csv",
                              index_col=0)
    df_out = df.merge(df_mask, on=['lon','lat'], how = 'inner')
    return df_out

def subset(df, year = 2000, month = 5):
    return df[(df['year'] == year) & (df['month'] == month)]

def timeToDays(timestr):
    if type(timestr) == str:
        days_span = re.search("days", timestr)
        days_str = timestr[:days_span.span()[0] - 1]
        hour_str = timestr[days_span.span()[1] + 1:days_span.span()[1] + 3]
        return int(days_str) if hour_str == '' else int(days_str) + round(int(hour_str) / 24)
    else:
        try:
            return int(timestr)
        except:
            return timestr


def make_map(df, variable, name='No name', legend='No legend'):
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    # print(geometry)
    var = df[variable]
    data = gpd.GeoDataFrame(var, crs="EPSG:4326", geometry=geometry)
    data = get_gdf_with_squares(data, delta_size=0.25)
    data['geoid'] = data.index.astype(str)

    m = folium.Map(location=[0, 0], tiles='cartodbpositron', zoom_start=2, control_scale=True)

    folium.Choropleth(
        geo_data=data,
        name=name,
        data=data,
        columns=['geoid', variable],
        key_on='feature.id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        line_color='white',
        line_weight=0,
        highlight=False,
        smooth_factor=1.0,
        # threshold_scale=[100, 250, 500, 1000, 2000],
        legend_name=legend).add_to(m)

    # add tooltips
    folium.features.GeoJson(data,
                            name='Labels',
                            style_function=lambda x: {'color': 'transparent', 'fillColor': 'transparent', 'weight': 0},
                            tooltip=folium.features.GeoJsonTooltip(fields=[variable],
                                                                   aliases=[variable + ': '],
                                                                   labels=True,
                                                                   sticky=False
                                                                   )
                            ).add_to(m)
    return m
