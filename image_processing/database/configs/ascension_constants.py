from image_processing.processing.image_processing import HSVRange

# elite
elite_hsv_range = HSVRange(hue_min=114, saturation_min=78, value_min=150,
                           hue_max=143, saturation_max=244, value_max=255)
# legendary
legendary_hsv_range = HSVRange(hue_min=16, saturation_min=101, value_min=120,
                               hue_max=36, saturation_max=234, value_max=255)
# all but ascended
mythic_hsv_range = HSVRange(hue_min=0, saturation_min=100, value_min=112,
                            hue_max=179, saturation_max=216, value_max=230)
# ascended
ascended_hsv_range = HSVRange(hue_min=88, saturation_min=0, value_min=187,
                              hue_max=155, saturation_max=132, value_max=255)

# The HSV ranges in this list are in order based on how many ascension classes
#   can by discovered by running an image through the filter
ALL_ASCENSION_HSV_RANGE = [elite_hsv_range, legendary_hsv_range,
                           ascended_hsv_range, mythic_hsv_range]
