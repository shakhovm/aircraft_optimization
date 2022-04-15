def pressure2alt(pressure):
    return 44331.5 - 4946.62 * (pressure * 100) ** 0.190263  # / 0.3048


def feet2meter(feet):
    meter = feet * 0.3048
    return meter


def meter2feet(meter):
    return meter / 0.3048
