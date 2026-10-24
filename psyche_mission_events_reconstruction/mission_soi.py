"""Current mission SOI choices, independent of SPICE and planetary sampling."""

from psyche_mission_events_reconstruction.paths import DATA_ROOT, resolve_input
from pathlib import Path
from .soi_events import LaplaceSOI

AU_M = 149597870700.0
FIXED_AXES_AU = {399: 1.0, 499: 1.52371034}
EARTH_EMB_SENSITIVITY_AU = 1.00000261
AXIS_SOURCE = "https://ssd.jpl.nasa.gov/planets/approx_pos.html"
ACTIVE_STAGE = DATA_ROOT / "soi"
ACTIVE_MARS_BOUNDARY = ACTIVE_STAGE / "mars_entry_result.json"


def fixed_mission_soi(body, mu_values):
    """Earth nominal 1 au; Mars frozen JPL Table 1 J2000 coefficient.

    JPL's 1.00000261 au is for the Earth-Moon barycentre, not Earth's
    geocentre. It is retained as a labelled sensitivity case, not used to
    tune the selected Earth radius to the SPK segment transition.
    """
    return LaplaceSOI(mu_values[str(body)], mu_values["10"], FIXED_AXES_AU[body] * AU_M)
