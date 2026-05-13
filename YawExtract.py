from PIL import Image
import re


def extract_xmp_pure_python(image_path):
    with Image.open(image_path) as img:
        # Extract the raw XMP string from image info
        xmp_raw = img.info.get('xmp') or img.applist[1][1]  # Varies by image type

    # Use Regex to find values in the XML string
    yaw = re.findall(r'FlightYawDegree="([^"]+)"', str(xmp_raw))
    alt = re.findall(r'RelativeAltitude="([^"]+)"', str(xmp_raw))

    return yaw[0], alt[0]

