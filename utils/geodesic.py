from geographiclib.geodesic import Geodesic
import numpy as np
from typing import List

geod = Geodesic.WGS84  # define the WGS84 ellipsoid


class Location:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def __str__(self):
        return f'Latitude: {self.latitude}, Longitude: {self.longitude}'

    def __repr__(self):
        return f'Latitude: {self.latitude}, Longitude: {self.longitude}'


def distance_to_degree(distance):
    return distance / geod.a


def bearing2coords(bearing, lat, lon, distance):
    bearing = np.deg2rad(bearing)
    la1 = np.deg2rad(lat)
    lo1 = np.deg2rad(lon)
    Ad = distance_to_degree(distance)
    la2 = np.arcsin(np.sin(la1) * np.cos(Ad) + np.cos(la1) * np.sin(Ad) * np.cos(bearing))
    lo2 = lo1 + np.arctan2(np.sin(bearing) * np.sin(Ad) * np.cos(la1), np.cos(Ad) - np.sin(la1) * np.sin(la2))
    x1 = np.rad2deg(la2)
    y1 = np.rad2deg(lo2)
    return x1, y1


def generate_new_route(waypoints: List[Location], deviation, distance):
    end_wp = waypoints[-1]
    new_route = []
    for start_wp in waypoints:
        bearing = geod.Inverse(start_wp.latitude, start_wp.longitude, end_wp.latitude, end_wp.longitude)['azi1']
        new_bearing = bearing + deviation
        new_start = Location(*bearing2coords(new_bearing, start_wp.latitude, start_wp.longitude, distance))
        new_route.append(new_start)
    return new_route


def geodesic_line_waypoints(loc_1, loc_2, n_waypoints):
    all_points = []
    l = geod.InverseLine(loc_1.latitude, loc_1.longitude, loc_2.latitude, loc_2.longitude)
    ds = l.s13 / (n_waypoints - 1)
    for i in range(n_waypoints - 1):
        s = min(ds * i, l.s13)
        g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        all_points.append(Location(g['lat2'], g['lon2']))
    all_points.append(Location(loc_2.latitude, loc_2.longitude))
    return all_points
