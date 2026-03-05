import numpy as np
from pathlib import Path

# Common senses.
SECOND_TO_NANOSECOND = 1_000_000_000

# Common dtype constants for waveform records.
TIME_DTYPE = np.int64
LENGTH_DTYPE = np.int64
CHANNEL_DTYPE = np.int16
DATA_DTYPE = np.dtype("f4")
INDEX_DTYPE = np.int32

# Placeholder index for not found items (will raise IndexError if used).
NOT_FOUND_INDEX = 999_999_999

# Baseline monitor interval.
N_BASELINE_MONITOR_INTERVAL = 100

# Pulse template with sampling rate of 38 kHz.
PULSE_TEMPLATE_38kHz = np.load(Path(__file__).parent / "msc" / "pulse_template_38kHz.npy")
PULSE_TEMPLATE_LENGTH = len(PULSE_TEMPLATE_38kHz)
PULSE_TEMPLATE_ARGMAX = np.argmax(PULSE_TEMPLATE_38kHz)

# Hit waveform recording window length, from the maximum of the hit waveform.
HIT_WINDOW_LENGTH_LEFT = 200
HIT_WINDOW_LENGTH_RIGHT = 600

# Energy resolution constants for truth generation.
# Reference photon energy and dx values for 25um wavelength
PHOTON_25um_meV = 49.62  # meV
PHOTON_25um_DX = 1.54e-6  # dx units
# Energy resolution in dx units (optimistic and conservative modes)
DX_RESOL_OPTIMISTIC = 186835.48206306322e-12
DX_RESOL_CONSERVATIVE = 267423.0098878706e-12

# Noise PSD
# Assumed (of_window_left, of_window_right) = (100, 300) samples
# This is a placeholder for the noise PSD,
# and only used when we cannot compute the noise PSD from the noise bank in a data-driven way.
NOISE_PSD_38kHz = np.load(Path(__file__).parent / "msc" / "noise_psd_38kHz.npy").tolist()

# Default path to template interpolation file
# This constructs path relative to this module's location
DEFAULT_TEMPLATE_INTERP_PATH = str(Path(__file__).parent / "msc" / "template_interp.pkl")
TEMPLATE_INTERP_FOLDER = str(Path(__file__).parent / "msc" / "sr2_pt2_templates")
