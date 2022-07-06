import folium
from shapely.geometry import Point
import shapely
import geopandas as gpd


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
