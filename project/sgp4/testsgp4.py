from sgp4.api import Satrec, jday, SGP4_ERRORS
import xml.etree.ElementTree as ET
from datetime import datetime

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

    try:
        ecc = float(ecc)
        if not (0.0 <= ecc <= 1.0):
            raise ValueError(f"Eccentricity value {ecc} is outside the valid range [0, 1].")
    except ValueError as e:
        print(f"Error processing eccentricity: {e}")
        return None, None

    try:
        mean_motion = float(mean_motion)
        incl = float(incl)
        ra_asc_node = float(ra_asc_node)
        arg_peri = float(arg_peri)
        mean_anom = float(mean_anom)
    except ValueError as e:
        print(f"Error processing TLE component: {e}")
        return None, None

    # Debugging output for all key values
    print(f"Debug - Eccentricity: {ecc}, Mean Motion: {mean_motion}, Inclination: {incl}, RA of Ascending Node: {ra_asc_node}, Arg of Periapsis: {arg_peri}, Mean Anomaly: {mean_anom}")

    ecc_str = f"{int(ecc * 1e7):07d}"
    print(f"Debug - Formatted Eccentricity in TLE: {ecc_str}")

    tle_line1 = (f"1 {norad_id:<5}U {intl_designator[:8]:<8} {str(epoch_year)[2:]:0>2}{epoch_day_fraction:012.8f} "
                 f"{float(mean_motion_dot):>10.8e} {float(mean_motion_ddot):>10.8e} 0 {int(float(bstar) * 1e5):05d}")

    tle_line2 = (f"2 {norad_id} {incl:8.4f} {ra_asc_node:8.4f} {ecc_str} "
                 f"{arg_peri:8.4f} {mean_anom:8.4f} {mean_motion:11.8f}")

    # Print the generated TLE lines for debugging
    print(f"Generated TLE Line 1: {tle_line1}")
    print(f"Generated TLE Line 2: {tle_line2}")
    
    return tle_line1, tle_line2

def convert_epoch_to_epoch_day_fraction(epoch):
    try:
        dt = datetime.strptime(epoch, '%Y-%m-%dT%H:%M:%S.%f')
        day_of_year = dt.timetuple().tm_yday + (dt.hour / 24.0) + (dt.minute / 1440.0) + (dt.second / 86400.0)
        year = dt.year
        return year, day_of_year
    except ValueError as e:
        print(f"Error parsing epoch: {e}")
        return None, None

def calculate_velocity_and_position(tle_line1, tle_line2):
    if tle_line1 is None or tle_line2 is None:
        return None, None
    
    try:
        year_str = tle_line1[18:20]
        day_of_year_str = tle_line1[20:32].strip()

        year = int("20" + year_str)
        day_of_year = float(day_of_year_str)
    except ValueError as e:
        print(f"Error parsing TLE date components: {e}")
        print(f"TLE Line 1: {tle_line1}")
        raise

    jd, fr = jday(year, 1, 0, 0, 0, 0)
    jd += day_of_year

    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    error_code, position, velocity = satellite.sgp4(jd, fr)
    
    if error_code == 0:
        print(f"Debug - SGP4 Position: {position}, Velocity: {velocity}")
        return position, velocity
    else:
        print(f"SGP4 Error Code: {error_code}, Description: {SGP4_ERRORS[error_code]}")
        print(f"TLE Line 1: {tle_line1}")
        print(f"TLE Line 2: {tle_line2}")
        print(f"Eccentricity Value Passed: {tle_line2[26:33].strip()}")
        raise RuntimeError(SGP4_ERRORS[error_code])

def test_hardcoded_tle():
    # A known valid TLE for testing purposes (example from Celestrak)
    tle_line1 = "1 25544U 98067A   21275.51902778  .00002182  00000-0  40810-4 0  9993"
    tle_line2 = "2 25544  51.6433 172.6192 0003345  49.0116 311.2654 15.48940629272085"

    calculate_velocity_and_position(tle_line1, tle_line2)

def main():
    # First, test with a hardcoded TLE to ensure SGP4 is working
    print("Testing with hardcoded TLE:")
    test_hardcoded_tle()

    # Now parse and generate TLE from XML file
    print("\nTesting with generated TLE from XML:")
    tle_data = parse_tle_xml('LEOxml.xml')
    for tle_line1, tle_line2 in tle_data:
        if tle_line1 is None or tle_line2 is None:
            continue
        position, velocity = calculate_velocity_and_position(tle_line1, tle_line2)
        if position and velocity:
            print(f"TLE Line 1: {tle_line1}")
            print(f"TLE Line 2: {tle_line2}")
            print(f"Position (km): {position}")
            print(f"Velocity (km/s): {velocity}\n")

if __name__ == "__main__":
    main()
