from typing import Dict, Any, List
from immutabledict import immutabledict
import strax
import straxion

common_options: Dict[str, Any] = dict(
    register_all=[straxion.plugins],
    register=[],
    check_available=(),
    store_run_fields=("name", "start", "end", "livetime", "mode"),
)

common_config = dict(
    n_dish_channels=4,  # KIDs inside the dish region
    n_side_channels=6,  # KIDs outside the dish region
    channel_map=immutabledict(
        # (Mimimum channel, maximum channel)
        # Channels must be listed in an ascending order
        kids=(0, 9),  # KIDs in the detector
    ),
)


def qualiphide(
    output_folder: str = "./strax_data",
    we_are_the_daq: bool = False,
    _processed_paths: List[str] = [],
    **kwargs: Any,
):
    """QUALIPHIDE base context for processing and analysis.

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

    st = strax.Context(config=common_config, **context_options)

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
