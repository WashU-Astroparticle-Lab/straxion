from typing import Dict, Any, List
from immutabledict import immutabledict
import strax
import straxion
from pathlib import Path
from .calibration import file_sha256, prepare_offline_calibration
from .offline import OfflineContext

common_options: Dict[str, Any] = dict(
    # register_all=[straxion.plugins],
    # register=[],
    check_available=(),
    store_run_fields=("name", "start", "end", "livetime", "mode"),
)

common_config = dict(
    channel_map=immutabledict(
        # (Mimimum channel, maximum channel)
        # Channels must be listed in an ascending order
        kids=(0, 40),  # KIDs in the detector
    ),
)

test_context_config = dict(
    n_dish_channels=4,  # KIDs inside the dish region
    n_side_channels=6,  # KIDs outside the dish region
    channel_map=immutabledict(
        # (Mimimum channel, maximum channel)
        # Channels must be listed in an ascending order
        kids=(0, 9),  # KIDs in the detector
    ),
)


def qualiphide_thz_offline(
    output_folder: str = "./strax_data",
    we_are_the_daq: bool = False,
    _processed_paths: List[str] = [],
    *,
    input_folder=None,
    calibration_dir=None,
    calibration_manifest=None,
    calibration_mode="averaged",
    calibration_options=None,
    **kwargs: Any,
):
    """QUALIPHIDE THz context for completed-run offline production and analysis.

    Averaged maps are required by default when accessing records or descendants.
    Supplying input_folder prepares missing calibration once; subsequent calls
    reuse its frozen manifest. Explicit per-scan processing remains available.

    Args:
        output_folder (str, optional): Path of the strax.DataDirectory where new data can be stored.
            Defaults to "./strax_data".
        we_are_the_daq (bool, optional): Whether this context runs on the DAQ machine.
            Defaults to False.
        _processed_paths (List[str], optional): Common paths of output data. Defaults to [].
        input_folder: Completed science-run directory containing timestream and scan npy files.
        calibration_dir: New map release directory; defaults to output_folder/calibration.
        calibration_manifest: Existing manifest.json, with input_folder specifying local data paths.
        calibration_mode: "averaged" (default) or explicit "per_scan" for legacy processing.
        calibration_options: Keyword arguments for prepare_offline_calibration, only on first use.
        config: Plugin configuration, including the science run's template_interp_folder.

    Returns:
        strax.Context: strax context object for processing.

    """
    if calibration_mode not in ("averaged", "per_scan"):
        raise ValueError("calibration_mode must be 'averaged' or 'per_scan'")
    if calibration_manifest is not None and calibration_options:
        raise ValueError("calibration_options apply only when preparing a new calibration")
    if calibration_manifest is not None and calibration_dir is not None:
        raise ValueError("Specify calibration_manifest or calibration_dir, not both")
    if input_folder is None and (calibration_dir is not None or calibration_options):
        raise ValueError("Calibration preparation options require input_folder")
    if input_folder is not None and calibration_mode == "per_scan":
        raise ValueError("Automatic completed-run preparation requires averaged mode")
    if input_folder is not None and calibration_manifest is None:
        directory = Path(calibration_dir or Path(output_folder) / "calibration").resolve()
        calibration_manifest = directory / "manifest.json"
        if not calibration_manifest.exists():
            calibration_manifest = prepare_offline_calibration(
                input_folder, directory, **(calibration_options or {})
            )
        elif calibration_options:
            raise ValueError("Calibration already frozen; omit options or use a new directory")
    if calibration_manifest is not None and input_folder is None:
        raise ValueError("calibration_manifest requires input_folder for scan/waveform locations")
    context_options = {**common_options, **kwargs}
    config = {**common_config, **context_options.pop("config", {})}
    context_options.update(
        calibration_mode=calibration_mode,
        calibration_manifest=(
            str(Path(calibration_manifest).resolve()) if calibration_manifest else ""
        ),
        calibration_manifest_sha256=(
            file_sha256(calibration_manifest) if calibration_manifest else ""
        ),
        science_run_input=str(Path(input_folder).resolve()) if input_folder is not None else "",
    )
    st = OfflineContext(config=config, **context_options)
    st.register(straxion.plugins.raw_records.QUALIPHIDETHzReader)
    st.register(straxion.plugins.records.DxRecords)
    st.register(straxion.plugins.hits.DxHits)
    st.register(straxion.plugins.hit_classification.DxHitClassification)
    st.register(straxion.plugins.substrate_events.SubstrateEvents)
    st.register(straxion.plugins.noise_bank.NoiseBank)
    st.register(straxion.plugins.truth.Truth)
    st.register(straxion.plugins.match.Match)

    # Add the output folder to the storage. This is where new data can be stored.
    st.storage = [
        strax.DataDirectory(output_folder, readonly=False),
    ]
    # Add the processed data to the storage.
    for path in _processed_paths:
        st.storage.append(
            strax.DataDirectory(
                path,
                readonly=False if we_are_the_daq else True,
            )
        )

    return st


def qualiphide_thz_online(
    output_folder: str = "./strax_data",
    we_are_the_daq: bool = False,
    _processed_paths: List[str] = [],
    **kwargs: Any,
):
    """QUALIPHIDE THz context for online processing.

    Args:
        output_folder (str, optional): Path of the strax.DataDirectory where new data can be stored.
            Defaults to "./strax_data".
        we_are_the_daq (bool, optional): Whether this context runs on the DAQ machine.
            Defaults to False.
        _processed_paths (List[str], optional): Common paths of output data. Defaults to [].

    Returns:
        strax.Context: strax context object for processing.

    """
    context_options = {**common_options, **kwargs}

    st = strax.Context(config=common_config, **context_options)
    st.register(straxion.plugins.raw_records.QUALIPHIDETHzReader)
    st.register(straxion.plugins.records.PulseProcessing)
    st.register_all(straxion.plugins.baseline_monitor)
    st.register(straxion.plugins.hits.Hits)
    st.register(straxion.plugins.hit_classification.HitClassification)

    # Add the output folder to the storage. This is where new data can be stored.
    st.storage = [
        strax.DataDirectory(output_folder, readonly=False),
    ]
    # Add the processed data to the storage.
    for path in _processed_paths:
        st.storage.append(
            strax.DataDirectory(
                path,
                readonly=False if we_are_the_daq else True,
            )
        )

    return st


def qualiphide(
    output_folder: str = "./strax_data",
    we_are_the_daq: bool = False,
    _processed_paths: List[str] = [],
    **kwargs: Any,
):
    """QUALIPHIDE test context for processing and analysis.

    Args:
        output_folder (str, optional): Path of the strax.DataDirectory where new data can be stored.
            Defaults to "./strax_data".
        we_are_the_daq (bool, optional): Whether this context runs on the DAQ machine.
            Defaults to False.
        _processed_paths (List[str], optional): Common paths of output data. Defaults to [].

    Returns:
        strax.Context: strax context object for processing and analysis.

    """
    context_options = {**common_options, **kwargs}

    st = strax.Context(config=test_context_config, **context_options)
    st.register(straxion.plugins.raw_records.NX3LikeReader)
    st.register(straxion.plugins.records.PulseProcessing)
    st.register_all(straxion.plugins.baseline_monitor)
    st.register(straxion.plugins.hits.Hits)
    st.register(straxion.plugins.hit_classification.HitClassification)

    # Add the output folder to the storage. This is where new data can be stored.
    st.storage = [
        strax.DataDirectory(output_folder, readonly=False),
    ]
    # Add the processed data to the storage.
    for path in _processed_paths:
        st.storage.append(
            strax.DataDirectory(
                path,
                readonly=False if we_are_the_daq else True,
            )
        )

    return st
