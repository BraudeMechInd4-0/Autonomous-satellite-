import xml.etree.ElementTree as ET
from sgp4.api import Satrec, jday
import datetime
import math

# Constants
EARTH_RADIUS_KM = 6378.137  # WGS84 Earth radius at the equator in kilometers
FLATTENING = 1 / 298.257223563  # WGS84 flattening of the Earth

def convert_epoch_to_epoch_day_fraction(epoch_str):
    timestamp_dt = datetime.datetime.fromisoformat(epoch_str)
    year = timestamp_dt.year
    day_of_year = timestamp_dt.timetuple().tm_yday
    fraction_of_day = (timestamp_dt.hour * 3600 + timestamp_dt.minute * 60 + timestamp_dt.second + timestamp_dt.microsecond / 1e12) / 86400.0
    epoch_day_fraction = day_of_year + fraction_of_day
    return year, epoch_day_fraction

def parse_tle_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    tle_data = []
    for segment in root.findall('.//segment'):
        tle_line1, tle_line2 = generate_tle_lines(segment)
        tle_data.append((tle_line1, tle_line2))
    return tle_data

def generate_tle_lines(segment):
    norad_id = segment.findtext('.//NORAD_CAT_ID').strip()
    classification = segment.findtext('.//CLASSIFICATION_TYPE').strip()
    intl_designator = segment.findtext('.//OBJECT_ID').strip().replace('-', '')

    epoch = segment.findtext('.//EPOCH').strip()
    mean_motion = segment.findtext('.//MEAN_MOTION').strip()
    ecc = segment.findtext('.//ECCENTRICITY').strip()
    incl = segment.findtext('.//INCLINATION').strip()
    ra_asc_node = segment.findtext('.//RA_OF_ASC_NODE').strip()
    arg_peri = segment.findtext('.//ARG_OF_PERICENTER').strip()
    mean_anom = segment.findtext('.//MEAN_ANOMALY').strip()
    bstar = segment.findtext('.//BSTAR').strip()
    mean_motion_dot = segment.findtext('.//MEAN_MOTION_DOT').strip()
    mean_motion_ddot = segment.findtext('.//MEAN_MOTION_DDOT').strip()

    epoch_year, epoch_day_fraction = convert_epoch_to_epoch_day_fraction(epoch)

    tle_line1 = (f"1 {norad_id}U {intl_designator[:8]:<8} {str(epoch_year)[2:]:0>2}{epoch_day_fraction:012.8f} "
                 f"{float(mean_motion_dot): .8e} {float(mean_motion_ddot): .8e} 0 {int(float(bstar) * 1e5):05d}")

    tle_line2 = (f"2 {norad_id} {float(incl):8.4f} {float(ra_asc_node):8.4f} {int(float(ecc) * 1e7):07d} "
                 f"{float(arg_peri):8.4f} {float(mean_anom):8.4f} {float(mean_motion):11.8f}")

    return tle_line1, tle_line2

def eci_to_geodetic(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    
    alt = r - EARTH_RADIUS_KM
    return lat, lon, alt

def compute_position_velocity_and_geodetic(tle_data, year, month, day, hour, minute, second):
    jd, fr = jday(year, month, day, hour, minute, second)
    results = []
    
    for tle_line1, tle_line2 in tle_data:
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        e, r, v = satellite.sgp4(jd, fr)
        if e == 0:
            # Calculate Euclidean distance
            distance_from_center = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
            distance_from_surface = distance_from_center - EARTH_RADIUS_KM
            
            # Convert ECI to Geodetic (Latitude, Longitude, Altitude)
            lat, lon, alt = eci_to_geodetic(r[0], r[1], r[2])
            
            results.append({
                'tle_line1': tle_line1,
                'tle_line2': tle_line2,
                'position': {'x': r[0], 'y': r[1], 'z': r[2]},
                'velocity': {'vx': v[0], 'vy': v[1], 'vz': v[2]},
                'distance_from_surface': distance_from_surface,
                'geodetic': {'latitude': lat, 'longitude': lon, 'altitude': alt}
            })
        else:
            results.append({
                'tle_line1': tle_line1,
                'tle_line2': tle_line2,
                'error': e
            })
    return results

# Parse TLE data from XML file
file_path = 'LEOxml.xml'  # Update this path if your XML file is in a different location
tle_data = parse_tle_xml(file_path)

# Define the time at which to compute the position and velocity
year, month, day, hour, minute, second = 2024, 8, 15, 7, 4, 54  # Set the exact time for the computation

# Compute position, velocity, and geodetic coordinates for each satellite
results = compute_position_velocity_and_geodetic(tle_data, year, month, day, hour, minute, second)

# Print the results
for result in results:
    if 'error' in result:
        print(f"Error for satellite with TLE:\n{result['tle_line1']}\n{result['tle_line2']}\nError code: {result['error']}")
    else:
        print(f"Satellite with TLE:\n{result['tle_line1']}\n{result['tle_line2']}")
        print(f"Position (km): x={result['position']['x']:.2f}, y={result['position']['y']:.2f}, z={result['position']['z']:.2f}")
        print(f"Velocity (km/s): vx={result['velocity']['vx']:.5f}, vy={result['velocity']['vy']:.5f}, vz={result['velocity']['vz']:.5f}")
        print(f"Distance from Earth's surface (km): {result['distance_from_surface']:.2f}")
       
