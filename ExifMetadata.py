from exif import Image


def dms_to_decimal(coords, ref):
    degrees = coords[0]
    minutes = coords[1]
    seconds = coords[2]

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

    # If South or West, the coordinate must be negative
    if ref in ['S', 'W']:
        decimal = -decimal

    return decimal


# Usage with your image
def extrat_lat_lon(image):
    with open(image, "rb") as f:
        img = Image(f)

    if img.has_exif:
        try:
            lat_decimal = dms_to_decimal(img.gps_latitude, img.gps_latitude_ref)
            lon_decimal = dms_to_decimal(img.gps_longitude, img.gps_longitude_ref)

            print(f"Decimal Latitude: {lat_decimal}")
            print(f"Decimal Longitude: {lon_decimal}")
            return lat_decimal, lon_decimal
        except AttributeError:
            print("GPS data not found in EXIF.")


lat_orig, lon_orig = extrat_lat_lon("DJI_0451.jpg")

