import straxion
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path
from scipy.signal import welch


def parse_arguments():
    """Parse command line arguments for the online monitor script."""
    parser = argparse.ArgumentParser(
        description="Generate online monitoring plots for straxion data analysis"
    )
    parser.add_argument(
        "--daq-input-dir", required=True, help="Path to the DAQ input file (.npy format)"
    )
    parser.add_argument(
        "--iq-finescan-dir", required=True, help="Directory containing IQ finescan files"
    )
    parser.add_argument(
        "--iq-finescan-filename",
        required=True,
        help="Filename of the IQ finescan file (.npy format)",
    )
    parser.add_argument(
        "--output-dir",
        default="online_monitor",
        help="Output directory for plots (default: online_monitor)",
    )
    parser.add_argument(
        "--mirror-channels",
        nargs="+",
        type=int,
        default=[1, 17, 28, 36],
        help="Mirror channels to highlight in plots (default: 1 17 28 36)",
    )

    return parser.parse_args()


def compute_psd_fft(data, sampling_rate=38_000):
    """
    Compute power spectral density using direct FFT
    """
    # Compute FFT
    fft_data = np.fft.fft(data)

    # Compute power spectral density
    psd = np.abs(fft_data) ** 2 / (len(data) * sampling_rate)

    # Create frequency array
    frequencies = np.fft.fftfreq(len(data), 1 / sampling_rate)

    # Take only positive frequencies
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    psd = psd[positive_freq_idx]

    # Double the power for positive frequencies (except DC and Nyquist)
    psd[1:-1] *= 2

    return frequencies, psd


def compute_psd_welch(data, sampling_rate=38_000, nperseg=None):
    """
    Compute power spectral density using Welch's method

    Parameters:
    data: 1D array of time series data
    sampling_rate: sampling frequency (1/t where t is time interval)
    nperseg: length of each segment for Welch method (default: len(data)//8)
    """
    if nperseg is None:
        nperseg = len(data) // 8

    frequencies, psd = welch(data, fs=sampling_rate, nperseg=nperseg)
    return frequencies, psd


def smooth_psd_log_bins(freq, psd, bins_per_decade=5, min_points=5):
    """
    Smooth PSD data using moving average with logarithmic scale binning.

    Parameters:
    -----------
    freq : array
        Frequency values
    psd : array
        Power spectral density values
    bins_per_decade : int
        Number of bins per decade in log scale
    min_points : int
        Minimum number of points required in each bin for averaging

    Returns:
    --------
    freq_smooth : array
        Smoothed frequency values (bin centers)
    psd_smooth : array
        Smoothed PSD values
    """
    # Remove zero frequencies and corresponding PSD values
    valid_idx = freq > 0
    freq_valid = freq[valid_idx]
    psd_valid = psd[valid_idx]

    if len(freq_valid) == 0:
        return np.array([]), np.array([])

    # Create logarithmic bins
    log_freq_min = np.log10(freq_valid.min())
    log_freq_max = np.log10(freq_valid.max())

    # Calculate number of bins
    n_decades = log_freq_max - log_freq_min
    n_bins = int(n_decades * bins_per_decade)

    # Create bin edges in log space
    log_bin_edges = np.linspace(log_freq_min, log_freq_max, n_bins + 1)
    bin_edges = 10**log_bin_edges

    # Initialize arrays for smoothed data
    freq_smooth = []
    psd_smooth = []

    # Process each bin
    for i in range(len(bin_edges) - 1):
        # Find points in current bin
        bin_mask = (freq_valid >= bin_edges[i]) & (freq_valid < bin_edges[i + 1])

        if np.sum(bin_mask) >= min_points:
            # Calculate bin center in log space
            log_center = (log_bin_edges[i] + log_bin_edges[i + 1]) / 2
            freq_center = 10**log_center

            # Calculate average PSD in this bin
            psd_avg = np.mean(psd_valid[bin_mask])

            freq_smooth.append(freq_center)
            psd_smooth.append(psd_avg)

    return np.array(freq_smooth), np.array(psd_smooth)


def generate_baseline_plot(baseline_monitor, run, mirror_channels, output_dir):
    """Generate baseline standard deviation plot."""
    plt.figure(dpi=200, figsize=(5, 3))

    # Plot mirror channels with labels
    for i in mirror_channels:
        plt.plot(
            baseline_monitor["baseline_monitor_interval"][0] / 1e9 * np.arange(100),
            baseline_monitor["baseline_monitor_std"][i, :],
            label=f"Channel {i}",
        )

    # Plot all channels with low alpha
    for i in range(41):
        plt.plot(
            baseline_monitor["baseline_monitor_interval"][0] / 1e9 * np.arange(100),
            baseline_monitor["baseline_monitor_std"][i, :],
            alpha=0.1,
            color="black",
            lw=0.5,
        )

    plt.xlabel("Time [Sec]")
    plt.ylabel("Baseline Std [rad]")
    plt.title(f"Run {run} Theta Baseline Std per 0.5 Seconds", fontsize=10)
    plt.legend(loc="best")
    plt.savefig(f"{output_dir}/baseline_std_{run}.png")
    plt.close()


def generate_amplitude_plot(hits, run, output_dir):
    """Generate amplitude distribution plot."""
    plt.figure(dpi=200, figsize=(5, 3))

    plt.hist(
        hits[~(hits["is_cr"] | hits["is_symmetric_spike"])]["amplitude_convolved_max_ext"],
        bins=np.linspace(0, 2.0, 100),
        label="Photon Candidates",
        histtype="step",
    )
    plt.hist(
        hits[hits["is_cr"]]["amplitude_convolved_max_ext"],
        bins=np.linspace(0, 2.0, 100),
        label="CR Hits",
        histtype="step",
    )
    plt.hist(
        hits[hits["is_symmetric_spike"]]["amplitude_convolved_max_ext"],
        bins=np.linspace(0, 2.0, 100),
        label="Symmetric Spikes",
        histtype="step",
    )

    plt.xlabel("Convolved Amplitude (Extended) [rad]")
    plt.ylabel("Counts")
    plt.legend(loc="best")
    plt.yscale("log")
    plt.title(f"Run {run} Convolved Amplitude (Extended)", fontsize=10)
    plt.savefig(f"{output_dir}/amplitudes_{run}.png")
    plt.close()


def generate_psd_plot(records, run, mirror_channels, output_dir):
    """Generate power spectral density plot."""

    plt.figure(dpi=200, figsize=(5, 3))

    # Plot all channels with low alpha
    for i in range(41):
        freq, psd = compute_psd_welch(records[i]["data_theta"])
        freq, psd = smooth_psd_log_bins(freq, psd)
        plt.semilogy(freq, psd, alpha=0.1, color="black", lw=0.5)

    # Plot mirror channels with labels
    for i in mirror_channels:
        freq, psd = compute_psd_welch(records[i]["data_theta"])
        freq, psd = smooth_psd_log_bins(freq, psd)
        plt.semilogy(freq, psd, label=f"Channel {i}")

    plt.ylabel(r"Power Spectral Density [rad$^2$/Hz]")
    plt.xscale("log")
    plt.title(f"Run {run} PSD with Welch Method and Log Bins Filtering", fontsize=10)
    plt.xlabel("Frequency (Hz)")
    plt.legend(loc="best")
    plt.savefig(f"{output_dir}/psd_{run}.png")
    plt.close()


if __name__ == "__main__":
    args = parse_arguments()

    # Initialize straxion context
    st = straxion.qualiphide_thz()

    # Extract run number from DAQ input filename
    run = Path(args.daq_input_dir).stem.split("-")[1]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare configuration
    configs = dict(
        daq_input_dir=args.daq_input_dir,
        iq_finescan_dir=args.iq_finescan_dir,
        iq_finescan_filename=args.iq_finescan_filename,
        symmetric_spike_min_slope=[200.0 for _ in range(41)],
    )

    print(f"Processing run {run}...")

    # Get data arrays
    print("Loading records...")
    records = st.get_array(run, "records", config=configs)

    print("Loading baseline monitor...")
    baseline_monitor = st.get_array(run, "baseline_monitor", config=configs)

    print("Loading hits...")
    hits = st.get_array(run, ("hits", "hit_classification"), config=configs)

    # Generate plots
    generate_baseline_plot(baseline_monitor, run, args.mirror_channels, args.output_dir)
    generate_amplitude_plot(hits, run, args.output_dir)
    generate_psd_plot(records, run, args.mirror_channels, args.output_dir)

    # Print done message
    print(f"Done for run {run}. Plots saved in {args.output_dir}/")
