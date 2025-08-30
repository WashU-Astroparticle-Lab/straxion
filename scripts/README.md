# Online Monitor Script

The `online_monitor.py` script generates monitoring plots for straxion data analysis, including baseline standard deviation, amplitude distributions, and power spectral density plots.

## Usage

```bash
python online_monitor.py --daq-input-dir <path> --iq-finescan-dir <path> --iq-finescan-filename <filename> [options]
```

## Required Arguments

- `--daq-input-dir`: Path to the DAQ input file (.npy format)
- `--iq-finescan-dir`: Directory containing IQ finescan files
- `--iq-finescan-filename`: Filename of the IQ finescan file (.npy format)

## Optional Arguments

- `--output-dir`: Output directory for plots (default: `online_monitor`)
- `--mirror-channels`: Mirror channels to highlight in plots (default: `1 17 28 36`)

## Example Usage

### Basic Usage
```bash
python online_monitor.py \
    --daq-input-dir "/path/to/ts_38kHz-1756457270.npy" \
    --iq-finescan-dir "/path/to/QLPHD/" \
    --iq-finescan-filename "iq_z_2dB_below_pcrit-1756457052.npy"
```

### Custom Output Directory
```bash
python online_monitor.py \
    --daq-input-dir "/path/to/ts_38kHz-1756457270.npy" \
    --iq-finescan-dir "/path/to/QLPHD/" \
    --iq-finescan-filename "iq_z_2dB_below_pcrit-1756457052.npy" \
    --output-dir "my_monitoring_plots"
```

### Custom Mirror Channels
```bash
python online_monitor.py \
    --daq-input-dir "/path/to/ts_38kHz-1756457270.npy" \
    --iq-finescan-dir "/path/to/QLPHD/" \
    --iq-finescan-filename "iq_z_2dB_below_pcrit-1756457052.npy" \
    --mirror-channels 5 10 15 20
```

## Example Case (Original Hardcoded Values)

The original script used these hardcoded values:

```bash
python online_monitor.py \
    --daq-input-dir "/Users/lanqingyuan/Documents/GitHub/straxion/.example_data/QLPHD/test/ts_38kHz-1756457270.npy" \
    --iq-finescan-dir "/Users/lanqingyuan/Documents/GitHub/straxion/.example_data/QLPHD/" \
    --iq-finescan-filename "iq_z_2dB_below_pcrit-1756457052.npy"
```

## Output

The script generates three plots in the specified output directory:

1. **Baseline Standard Deviation** (`baseline_std_{run}.png`): Shows baseline stability over time for all channels
2. **Amplitude Distribution** (`amplitudes_{run}.png`): Histogram of hit amplitudes categorized by type
3. **Power Spectral Density** (`psd_{run}.png`): Frequency analysis of the data using Welch's method

## Dependencies

- `straxion`: Main data analysis framework
- `matplotlib`: Plotting library
- `numpy`: Numerical computing
- `scipy`: Scientific computing (for PSD calculations)
- `pathlib`: Path manipulation utilities

## Notes

- The script automatically extracts the run number from the DAQ input filename
- All channels are processed, with mirror channels highlighted in the plots
- The output directory is created automatically if it doesn't exist
- Progress messages are displayed during execution
