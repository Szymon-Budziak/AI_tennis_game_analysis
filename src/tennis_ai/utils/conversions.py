__all__ = ['convert_pixels_to_meters_distances', 'convert_meters_to_pixels_distances']


def convert_pixels_to_meters_distances(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels


def convert_meters_to_pixels_distances(meter_distance, reference_height_in_meters, reference_height_in_pixels):
    return (meter_distance * reference_height_in_pixels) / reference_height_in_meters
