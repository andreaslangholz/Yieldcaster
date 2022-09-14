""" 
MSc. Project 2022 - Andreas Langholz - Geographical utilities
Description: Collection of functions related to geometry and mapping 
of pixels and outputting visuals
"""


import folium
from shapely.geometry import Point
import shapely
import geopandas as gpd
import numpy as np
from math import dist


def get_square_around_point(point_geom, delta_size=0.25):
    """ 
    Converts a point to a pixel square of 0.5 x 0.5 lon, lat 
    """
    point_coords = np.array(point_geom.coords[0])

    c1 = point_coords + [-delta_size, -delta_size]
    c2 = point_coords + [-delta_size, +delta_size]
    c3 = point_coords + [+delta_size, +delta_size]
    c4 = point_coords + [+delta_size, -delta_size]

    square_geom = shapely.geometry.Polygon([c1, c2, c3, c4])

    return square_geom

def get_gdf_with_squares(gdf_with_points, delta_size=0.25):
    """ 
    converts a full Geopandas df to squares from points
    """
    
    gdf_squares = gdf_with_points.copy()
    gdf_squares['geometry'] = (gdf_with_points['geometry']
                               .apply(get_square_around_point,
                                      delta_size))

    return gdf_squares


def make_map(df, variable, name='No name', legend='No legend', custom_scale = False):
    """ Converts a dataframe to a Folium map and overlay on an
    Openmaps API and shows the inputted variable parameter as a heatmap
    
    parameters:
    df: dataframe with lon, lat in the columns and a 'variable' column with 
    variable: the variable that shoudl be displayed in the heatmap
    """

    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    var = df[variable]
    data = gpd.GeoDataFrame(var, crs="EPSG:4326", geometry=geometry)
    data = get_gdf_with_squares(data, delta_size=0.25)
    data['geoid'] = data.index.astype(str)

    m = folium.Map(location=[0, 0], tiles='cartodbpositron', zoom_start=2, control_scale=True)
    
    # Custom scale, can be lefte out for depending on visual
    if custom_scale:
        custom_scale = (df[variable].quantile((0,0.25,0.5,0.75,1))).tolist() 
        folium.Choropleth(
            geo_data=data,
            name=name,
            data=data,
            columns=['geoid', variable],
            key_on='feature.id',
            fill_color='RdBu',
            fill_opacity=0.9,
            line_opacity=0.2,
            line_color='white',
            line_weight=0,
            highlight=False,
            smooth_factor=1.0,
            threshold_scale=custom_scale,
            legend_name=legend).add_to(m)
    else:
        folium.Choropleth(
                    geo_data=data,
                    name=name,
                    data=data,
                    columns=['geoid', variable],
                    key_on='feature.id',
                    fill_color='RdBu',
                    fill_opacity=0.9,
                    line_opacity=0.2,
                    line_color='white',
                    line_weight=0,
                    highlight=False,
                    smooth_factor=1.0,
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

def find_nearest(array, value):
    """ finds the two nearest values in array from input value
    (1 on each side of the value input)
    """ 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    x = array[idx]
    if x > value:
        y = array[idx - 1]
    else:
        y = array[idx + 1]
    return x, y

def convert(array_lon, array_lat, lon, lat):
    """ Calculate the euclidian distance to the four closest points
    and output the pointset with the lowest distance
    """
    p = (lon, lat)
    lon1, lon2 = find_nearest(array_lon, p[0])
    lat1, lat2 = find_nearest(array_lat, p[1])
    ar = np.array([[lon1, lat1, dist((lon1, lat1), p)],
                   [lon1, lat2, dist((lon1, lat2), p)],
                   [lon2, lat1, dist((lon2, lat1), p)],
                   [lon2, lat2, dist((lon2, lat2), p)]])
    ara = ar[ar[:, 2].argmin(), :]
    return ara[0], ara[1]

def geo_subset(df, area = 'all'):
    """ Subsets a dataframe to an area
    takes inputs ['all', 'europe', 'france', 'china', 'us']
    """
    if area == 'all':
        return df
    else:
        coord_france = {'min_lat': 42.5, 'max_lat': 50,
                        'min_lon': -2, 'max_lon': 7.5}
        coord_europe = {'min_lat': 35, 'max_lat': 58,
                        'min_lon': -13, 'max_lon': 35}
        coord_US = {'min_lat': 26, 'max_lat': 48, 'min_lon': -125, 'max_lon': -68}
        coord_china = {'min_lat': 1, 'max_lat': 44, 'min_lon': 77, 'max_lon': 122}
        coord = coord_china if area == 'china'else coord_france if area == 'france' else coord_europe if area == 'europe' else coord_US
        df_new = df[(df['lat'] > coord['min_lat']) & (df['lat'] < coord['max_lat']) & (
            df['lon'] > coord['min_lon']) & (df['lon'] < coord['max_lon'])]
        df_new = df_new.reset_index(drop = True)
        return df_new


