import geopandas as gpd
from shapely.geometry import Point
from raster2xyz.raster2xyz import Raster2xyz
import pandas as pd
from src.utils import get_data, rmse, baseline_rolling
import numpy as np
import shapely
import math

path_in = "C:\\Users\\Andreas Langholz\\Downloads\\whe_2010_har.tif"
path_out = "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\raw\\harvest_area\\whe_2010_har.csv"

rtxyz = Raster2xyz()
rtxyz.translate(path_in, path_out)

myRasterDF = pd.read_csv(path_out)

myRasterDF['x'].tail(50)


def round_to_quarters(x):
    x_abs = abs(x)
    dif = x_abs - round(x_abs)
    new_dec = 0.75 if dif < 0 else 0.25
    x_out = math.copysign(1,x) * (math.floor(x_abs) + new_dec)
    return x_out

myRasterDF['lon'] = myRasterDF['x'].apply(round_to_quarters)
myRasterDF['lat'] = myRasterDF['y'].apply(round_to_quarters)

df, _ = get_data("wheat_winter", 'pix')


p = Point(df.lon[0], df.lat[0])
geometry = [Point(xy) for xy in zip(df.lon, df.lat)]


for xy in zip(df.lon, df.lat):
    print(xy)

var = df['yield']

data = gpd.GeoDataFrame(var, crs="EPSG:4326", geometry=geometry)

print(data['geometry'][0])
p = data['geometry'][0]

df.lon
point_coords = np.array(p)
point_coords
delta_size = 0.25
c1 = point_coords + [-delta_size, -delta_size]
c2 = point_coords + [-delta_size, +delta_size]
c3 = point_coords + [+delta_size, +delta_size]
c4 = point_coords + [+delta_size, -delta_size]
c3

square_geom = shapely.geometry.Polygon([c1, c2, c3, c4])
print(square_geom)

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


def make_map(df, variable, name='No name', legend='No legend'):
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    # print(geometry)
    var = df[variable]
    data = gpd.GeoDataFrame(var, crs="EPSG:4326", geometry=geometry)
    data = get_gdf_with_squares(data, delta_size=0.25)
    data['geoid'] = data.index.astype(str)
