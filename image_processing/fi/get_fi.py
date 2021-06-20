import numpy as np
import image_processing.build_db as BD
import image_processing.processing as processing
import image_processing.globals as GV
import image_processing.scripts.getSISize as siScript


if __name__ == "__main__":
    imageDB = BD.buildDB(enrichedDB=True)

    hero_ss = GV.image_ss

    heroesDict, rows = processing.getHeroes(
        hero_ss, si_adjustment=0)

    digit_bins = {}
    for k, v in heroesDict.items():
        bins = siScript.getDigit(v["image"])
        v["digit_info"] = bins
        for digitName, tempDigitdict in bins.items():

            digitTuple = tempDigitdict["digit_info"]
            digitTop = digitTuple[0]
            digitBottom = digitTuple[1]
            digitHeight = digitBottom - digitTop
            if digitName not in digit_bins:
                digit_bins[digitName] = []
            digit_bins[digitName].append(digitHeight)
    avg_bin = {}
    total_digit_occurrences = 0
    for k, v in digit_bins.items():
        avg = np.mean(v)
        avg_bin[k] = {}
        avg_bin[k]["height"] = avg
        occurrence = len(v)
        avg_bin[k]["count"] = occurrence
        total_digit_occurrences += occurrence

        # print("{} {}".format(k, avg))
    graded_avg_bin = {}
    for si_name, image_dict in baseImages.items():
        if si_name not in graded_avg_bin:
            graded_avg_bin[si_name] = {}
        frequency_height_adjust = 0
        for digit_name, scale_dict in avg_bin.items():

            v_scale = baseImages[si_name][digit_name]["v_scale"]

            digit_count = scale_dict["count"]
            digit_height = scale_dict["height"]
            digit_freqency = digit_count / total_digit_occurrences

            frequency_height_adjust += (v_scale *
                                        digit_height) * digit_freqency
            # print(si_name, v_scale * digit_height)
            # graded_avg_bin[si_name]["height"] += graded_avg_bin[si_name][
            #     "height"] + frequency_height_adjust
        graded_avg_bin[si_name]["height"] = frequency_height_adjust
