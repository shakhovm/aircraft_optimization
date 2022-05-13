import json

from geographiclib.geodesic import Geodesic
import numpy as np

import openap
from openap.extra.aero import mach2tas
from utils.geodesic import Location, generate_new_route, bearing2coords, geodesic_line_waypoints
from utils.units_converter import feet2meter, meter2feet, pressure2alt, mperstokt

geod = Geodesic.WGS84  # define the WGS84 ellipsoid


class Aircraft:
    def __init__(self, name):
        self.name = name
        self.fuelflow = openap.FuelFlow(ac='A320', eng='CFM56-5B4')
        self.wrap = openap.WRAP(ac='A320')

    def fuel_enroute(self, gross_weight, tas, altitude, path_angle):
        """Compute the fuel flow during climb, cruise, or descent.
        The net thrust is first estimated based on the dynamic equation.
        Then FuelFlow.at_thrust() is called to compted the thrust. Assuming
        no flap deflection and no landing gear extended.
        Args:
            mass (int or ndarray): Aircraft mass (unit: kg).
            tas (int or ndarray): Aircraft true airspeed (unit: kt).
            alt (int or ndarray): Aircraft altitude (unit: ft).
            path_angle (float or ndarray): Flight path angle (unit: degrees).
        Returns:
            float: Fuel flow (unit: kg/s).
        """
        FF = self.fuelflow.enroute(mass=gross_weight, tas=tas, alt=altitude, path_angle=path_angle)
        return FF

    @property
    def cruise_mach(self):
        return self.wrap.cruise_mach()

    @property
    def cruise_alt(self):
        return self.wrap.cruise_alt()


class AircraftEnv:
    def __init__(self, arrival_location, destination, n_waypoints=9, n_routes=5,
                 estimated_time=29700, visualize=False):
        self.aicraft = Aircraft('A320')
        self.arrival_location = arrival_location
        self.destination = destination
        self.n_waypoints = n_waypoints
        self.n_routes = n_routes
        self.main_trajectory = geodesic_line_waypoints(arrival_location, destination, n_waypoints)

        self.waypoints = self._generate_alternative_routes()

        self.cruise_mach_range = np.arange(self.aicraft.cruise_mach['minimum'],
                                           self.aicraft.cruise_mach['maximum'], 0.01)
        self.cruise_alt_min = self.aicraft.cruise_alt['minimum'] * 1000

        self.cruise_alt_max = self.aicraft.cruise_alt['maximum'] * 1000

        self.estimated_time = estimated_time
        self._load_wind_info()
        self.state_info = None
        self.state = self.reset()

        self.visualize = visualize

    def _generate_alternative_routes(self):
        possible_angles = [270, 270, 90, 90]  # angles between course and new point (degrees)
        possible_distance = [100e3, 50e3, 50e3, 100e3]  # distance between point of main trajectory and new point
        waypoints = []
        for deviation, distance in zip(possible_angles, possible_distance):
            new_route = [self.main_trajectory[0]] + \
                        generate_new_route(self.main_trajectory[1:-1], deviation, distance) + \
                        [self.main_trajectory[-1]]
            waypoints.append(new_route)
        waypoints.insert(len(waypoints) // 2, self.main_trajectory)
        return waypoints

    def _load_wind_info(self, file_name="wind.json"):
        with open(file_name) as f:
            winds = json.load(f)
        self._wind_altitude = pressure2alt(np.array(winds['wind_pressures']))
        self._wind_magnitude = np.array(winds['wind']) * 5
        self._wind_direction = np.array(winds['wdir'])

    def _wind_interpolation(self, altitude, trajectory, waypoint):
        magnitude = np.interp(altitude, self._wind_altitude, self._wind_magnitude[:, trajectory, waypoint])
        direction = np.interp(altitude, self._wind_altitude, self._wind_direction[:, trajectory, waypoint])
        return magnitude, direction

    def _solver_wind_interpolation(self, m, altitude, trajectory, waypoint):
        magnitude = np.interp(altitude, self._wind_altitude, self._wind_magnitude[:, trajectory, waypoint])
        direction = np.interp(altitude, self._wind_altitude, self._wind_direction[:, trajectory, waypoint])
        return magnitude, direction

    def _waypoints_to_array(self):
        ar = []
        for trajectory in self.waypoints:
            ar.append([[wp.latitude, wp.longitude] for wp in trajectory])
        return np.array(ar)

    def ground_speed(self, tas, wind_speed, wind_direction, course):
        wind_correction_angle = np.arcsin((wind_speed / tas) *
                                          np.sin(course - wind_direction))
        ground_speed = np.sqrt(tas ** 2 + wind_speed ** 2 +
                               2 * (tas * wind_speed * np.cos(course - wind_direction + wind_correction_angle)))
        return ground_speed, wind_correction_angle

    @staticmethod
    def solver_ground_speed(m, tas, wind_speed, wind_direction, course):
        wind_correction_angle = m.asin((wind_speed / tas) *
                                          m.sin(course - wind_direction))
        ground_speed = m.sqrt(tas ** 2 + wind_speed ** 2 +
                               2 * (tas * wind_speed * m.cos(course - wind_direction + wind_correction_angle)))
        return ground_speed, wind_correction_angle

    def solver_step(self, action, m):
        self.state['altitude'] += action['altitude']

        wp_index = self.state['waypoint']
        self.state['waypoint'] += 1
        start_wp = self.waypoints[self.state['trajectory']][wp_index]
        end_wp = self.waypoints[action['trajectory']][wp_index + 1]

        # Get wind magnitude, direction for altitude
        wind_magnitude, wind_direction = 1, 1

        # s12 - distance, azi1 - bearing; Calculate ground speed
        two_points_info = geod.Inverse(start_wp.latitude, start_wp.longitude, end_wp.latitude, end_wp.longitude)
        print(self.state['altitude'])
        tas = mach2tas(action["mach_number"], self.state['altitude'])
        tas = 2000 * action['mach_number'] #mach2tas(action["mach_number"], self.state['altitude'])
        velocity, wind_correction_angle = self.solver_ground_speed(m, tas, wind_magnitude, np.deg2rad(wind_direction),
                                                            np.deg2rad(two_points_info['azi1']))

        fuel_flow = self.state['altitude'] * action['mach_number'] * 100 #self.aicraft.fuel_enroute(30000, tas, altitude, path_angle=0) *10**2

        time_for_distance = two_points_info['s12'] / velocity

        fuel_burn = fuel_flow * time_for_distance

        # Define reward
        reward = -fuel_burn

        self.state['trajectory'] = action['trajectory']
        return self.state, reward

    def reset(self):
        self.state = {
            "trajectory": len(self.waypoints) // 2,
            "waypoint": 0,
            "altitude": self.cruise_alt_min,
            "total_time": 0
        }

        self.state_info = {
            "trajectory": len(self.waypoints) // 2,
            "waypoint": 0,
            "speed": 0,
            "tas": 0,
            "distance": 0,
            "wind_magnitude": 0,
            "wind_direction": 0,
            "course": 0,
            "correction_angle": 0,
            "altitude": 0,
            "time_for_distance": 0,
            "fuel_burn": 0,
            "total_time": 0,
            "reward": 0,
            "fuel_flow": 0
        }
        return self.state

    def step(self, action, for_training=True):
        self.state['altitude'] += action['altitude']
        altitude = self.state['altitude']
        wp_index = self.state['waypoint']
        self.state['waypoint'] += 1
        start_wp = self.waypoints[self.state['trajectory']][wp_index]
        end_wp = self.waypoints[action['trajectory']][wp_index + 1]

        # Get wind magnitude, direction for altitude
        wind_magnitude, wind_direction = self._wind_interpolation(altitude, self.state['trajectory'], wp_index)

        # s12 - distance, azi1 - bearing; Calculate ground speed
        two_points_info = geod.Inverse(start_wp.latitude, start_wp.longitude, end_wp.latitude, end_wp.longitude)
        tas = mach2tas(action["mach_number"], self.state['altitude'])
        velocity, wind_correction_angle = self.ground_speed(tas, wind_magnitude, np.deg2rad(wind_direction),
                                                            np.deg2rad(two_points_info['azi1']))

        fuel_flow = self.aicraft.fuel_enroute(30000, mperstokt(tas), meter2feet(altitude),
                                              path_angle=0)

        time_for_distance = two_points_info['s12'] / velocity
        done = self.state['waypoint'] == self.n_waypoints - 1

        fuel_burn = fuel_flow * time_for_distance
        self.state['total_time'] += time_for_distance

        # Define reward
        reward = -fuel_burn

        # if self.state['altitude'] < self.cruise_alt_min or self.state['altitude'] > self.cruise_alt_max:
        #     reward += -40000
        #     if for_training:
        #         done = True

        # if abs(action['trajectory'] - self.state['trajectory']) > 1:
        #     reward += -10000
        #     done = True

        self.state['trajectory'] = action['trajectory']
        #
        # if done and self.state['total_time'] > self.estimated_time:
        #     reward += -1000 * (self.state['total_time'] - self.estimated_time)

        # Define state info
        self.state_info = {
            "trajectory": self.state['trajectory'],
            "waypoint": self.state['waypoint'],
            "speed": velocity,
            "tas": tas,
            "distance": two_points_info['s12'],
            "wind_magnitude": wind_magnitude,
            "wind_direction": wind_direction,
            "course": two_points_info['azi1'],
            "correction_angle": np.rad2deg(wind_correction_angle),
            "altitude": altitude,
            "time_for_distance": time_for_distance,
            "fuel_burn": fuel_burn,
            "total_time": self.state["total_time"],
            "fuel_flow": fuel_flow,

            "reward": reward
        }
        if self.visualize:
            print(f' Velocity {self.state_info["velocity"]}\n tas {self.state_info["tas"]}\n '
                  f'Distance {self.state_info["distance"]}\n '
                  f'Wind Magnitude/Direction {self.state_info["wind_magnitude"]}/'
                  f'{self.state_info["wind_direction"]}\n Course {self.state_info["course"]}\n '
                  f'Altitude {self.state_info["altitude"]}\n '
                  f'Time For Distance {self.state_info["time_for_distance"]}\n '
                  f'Fuel Burn {self.state_info["fuel_burn"]}\n Total Time {self.state["total_time"]}')

        return self.state, reward, done

    def rand_action(self):
        mach_number = np.random.choice(self.cruise_mach_range)
        altitude_actions = np.array([feet2meter(1000),
                                     feet2meter(2000),
                                     feet2meter(-1000),
                                     feet2meter(-2000),
                                     feet2meter(0)])
        altitude_actions = altitude_actions[(self.state['altitude'] + altitude_actions <= self.cruise_alt_max) &
                                            (self.state['altitude'] + altitude_actions >= self.cruise_alt_min)]

        altitude = np.random.choice(altitude_actions)
        return {
            "mach_number": mach_number,
            "trajectory": np.random.randint(0, self.n_routes),
            "altitude": altitude
        }

    def get_state_info(self):
        return self.state_info

    def __repr__(self):
        return "\n".join([f"{key} : {value}" for key, value in self.state_info.items()])

