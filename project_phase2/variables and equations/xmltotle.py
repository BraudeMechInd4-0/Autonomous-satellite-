import xml.etree.ElementTree as ET

def extract_tle_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract TLE parameters
    object_name = root.find(".//OBJECT_NAME").text
    norad_id = root.find(".//NORAD_CAT_ID").text
    epoch = root.find(".//EPOCH").text
    mean_motion_dot = root.find(".//MEAN_MOTION_DOT").text
    bstar = root.find(".//BSTAR").text
    inclination = root.find(".//INCLINATION").text
    ra_of_asc_node = root.find(".//RA_OF_ASC_NODE").text
    eccentricity = root.find(".//ECCENTRICITY").text.replace(".", "")
    arg_of_pericenter = root.find(".//ARG_OF_PERICENTER").text
    mean_anomaly = root.find(".//MEAN_ANOMALY").text
    mean_motion = root.find(".//MEAN_MOTION").text
    rev_at_epoch = root.find(".//REV_AT_EPOCH").text

    # Construct TLE lines
    line1 = f"1 {norad_id}U {epoch[:10].replace('-', '')} {mean_motion_dot[:7]:<8} 0  {bstar[:7]:<8} 0  9999"
    line2 = f"2 {norad_id} {inclination[:8]:<8} {ra_of_asc_node[:8]:<8} {eccentricity[:7]:<7} {arg_of_pericenter[:7]:<7} {mean_anomaly[:7]:<7} {mean_motion:<11} {rev_at_epoch:<6}"

    print(f"Object: {object_name}")
    print(line1)
    print(line2)

# File paths for each satellite
files = [
    'IRIDIUM33deb.xml',
    'QIANFAN4.xml',
    'SKYNET4C.xml',
    'STARLINK1341.xml',
    'ASBM2.xml'
]

# Extract and print TLE from each file
for file_path in files:
    extract_tle_from_xml(file_path)
