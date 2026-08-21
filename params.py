"""
Training parameters dataclass and serialization utilities.
"""

import dataclasses
import os
import random
from pathlib import Path
from typing import Literal

import json_utils


@dataclasses.dataclass
class Params:
    """Training and model configuration parameters."""

    # Data and mode parameters
    VARIABLE: Literal['tmax', 'precip'] = 'tmax'
    # Output distribution key. When None the distribution is inferred from
    # VARIABLE via model_factory.VARIABLE_TO_DISTRIBUTION (tmax->gaussian,
    # precip->bernoulli_gamma). Set it to override. Must be a key of
    # model_factory.LIKELIHOODS (validated in build_model, not here, to avoid a
    # circular import).
    DISTRIBUTION: str | None = None
    DATA_YEAR_START: int | None = 2023  # First year of data to load (inclusive), None = all years
    DATA_YEAR_END: int | None = None    # Last year of data to load (inclusive), None = no upper bound

    # Ablation parameters
    SEASONAL_FEATURES: bool = True  # Whether to include seasonal features (cos/sin of day-of-year) at all
    SEASONAL_FEATURES_IN_MLP: bool = True  # Whether to include seasonal features (cos/sin of day-of-year) in elevation MLP
    USE_ELEVATION: bool  = True  # Whether to include the elevation maps
    USE_MTPI: bool  = True  # Only used if USE_ELEVATION is also True
    # Whether to include the geopotential-derived coarse elevation channel as a
    # context (encoder input) channel in the surface configuration. This is
    # SEPARATE from USE_ELEVATION (which drives the DEM/TPI elevation-bias MLP):
    # setting this False drops only the geopotential input channel, while
    # geopotential is still loaded so the MLP's elev_diff feature keeps working.
    # Used for the "no-geopotential" surface ablation (CLI --no-geopotential).
    USE_ELEVATION_CHANNEL: bool = True

    # Input-source toggle: use EITHER the ERA5-Land surface field OR the ERA5
    # pressure-level (atmospheric) fields — exactly one, never both and never
    # neither (enforced in __post_init__). USE_SURFACE gates only the surface
    # variable channel; the lat/lon/seasonal/elevation scaffold is always present.
    USE_SURFACE: bool = True
    USE_ATMOSPHERIC: bool = False
    # Surface precipitation anchor (only used if USE_SURFACE). Appends the
    # ERA5-Land daily total precipitation field (tp, mm) as an extra dense context
    # channel alongside the surface t2m_max data channel — the surface analogue of
    # USE_SFC_ATMOS, and the coarse counterpart of a precip target. Same 29x61 grid
    # and daily axis as the surface field, so no regridding. Requires USE_SURFACE
    # (enforced in __post_init__).
    USE_SURFACE_PRECIP: bool = False
    # Atmospheric-field selection (only used if USE_ATMOSPHERIC). Each
    # (variable, level, hour) becomes its own context channel.
    ATMOS_VARIABLES: list[str] = dataclasses.field(
        default_factory=lambda: ['z', 't', 'q'])
    ATMOS_LEVELS: list[int] = dataclasses.field(
        default_factory=lambda: [1000, 925, 850, 700, 500, 300])
    ATMOS_HOURS: list[str] = dataclasses.field(
        default_factory=lambda: ['00', '06', '12', '15', '18'])
    # Surface anchor (only used if USE_ATMOSPHERIC). Optionally append ERA5
    # single-level surface field(s) (0.25 deg, same grid and hours as the
    # pressure levels) as extra dense context channels — a near-surface anchor
    # for the atmospheric configuration. Each (variable, hour) becomes its own
    # channel, appended after the pressure-level channels. Requires
    # USE_ATMOSPHERIC (enforced in __post_init__).
    USE_SFC_ATMOS: bool = False
    ATMOS_SFC_VARIABLES: list[str] = dataclasses.field(
        default_factory=lambda: ['t2m', 'tp'])
    # When USE_ATMOSPHERIC: feed the atmospheric field to the encoder at its
    # native coarse (0.25 deg) grid instead of interpolating it up to the fine
    # surface grid. The RBF final layer maps any grid -> target points, so the
    # whole stack is resolution-agnostic; this avoids the costly upsample and
    # keeps the atmospheric channels (and a coarse lat/lon/seasonal/elevation
    # scaffold) on their own grid. Only used if USE_ATMOSPHERIC=True.
    ATMOS_NATIVE_GRID: bool = True

    # Encoder selection. 'flat' is the original single depthwise SetConv over all
    # channels (unchanged behaviour). The 'grouped_*' encoders bucket channels by
    # CHANNEL_GROUP_BY and encode each labeled group to a shared position before
    # the CNN; see convCNP/models/grouped_encoder.py.
    ENCODER: Literal['flat', 'grouped_setconv', 'channel_attention'] = 'flat'
    CHANNEL_GROUP_BY: Literal['variable'] = 'variable'

    # Model parameters
    N_CHANNELS: int = 128    # default in paper is 128
    N_BLOCKS: int = 6        # default in paper is 6
    KERNEL_SIZE: int = 5     # default in paper is 5
    LENGTH_SCALE: float = 0.1  # default in paper is 0.1
    IN_CHANNELS: int = 0     # default in paper is 25; this is best computed dynamically from the data available

    # Training parameters
    N_EPOCHS: int = 30       # default in paper is 100
    # Monte-Carlo Gamma draws for the sample-based Bernoulli-Gamma CRPS loss
    # (distribution 'bernoulli_gamma_crps'); ignored for the NLL loss.
    CRPS_N_SAMPLES: int = 25
    BATCH_SIZE: int = 16     # default in paper is 16
    LR: float = 5e-4         # default in paper is 5e-4
    PATIENCE: int = 10       # default in paper is 10

    # Cross-validation parameters
    N_FOLDS: int = 5         # default in paper is 5

    # Other parameters
    SEED: int = 42

    # logging/output parameters
    TRIAL_NAME: str = 'chr_base-30-5folds'

    RUN_TYPE: Literal['local', 'cloud'] = 'local'

    DEVICE: str = 'cpu'  # Converted to torch.device in __post_init__

    # Data paths (saved during training for reproducibility)
    ERA5_MAX_TEMP_GLOB: str | None = None
    ERA5_PRECIP_GLOB: str | None = None
    ERA5_GEOPOTENTIAL_GLOB: str | None = None
    ERA5_PRESSURE_LEVEL_DIR: str | None = None
    ERA5_SURFACE_DIR: str | None = None
    METEO_SWISS_MAX_TEMP_GLOB: str | None = None
    METEO_SWISS_PRECIP_GLOB: str | None = None
    HI_RES_TOPOGRAPHY_ZARR_PATH: str | None = None

    def __post_init__(self):
        assert self.N_EPOCHS > 0, "N_EPOCHS must be positive"
        assert self.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert self.N_FOLDS > 0, "N_FOLDS must be positive"
        assert self.LR > 0, "LR must be positive"
        assert self.VARIABLE in ('tmax', 'precip'), f"Unknown VARIABLE: {self.VARIABLE}"
        assert self.RUN_TYPE in ('local', 'cloud'), f"Unknown RUN_TYPE: {self.RUN_TYPE}"
        assert self.USE_SURFACE != self.USE_ATMOSPHERIC, \
            "Exactly one of USE_SURFACE / USE_ATMOSPHERIC must be True " \
            "(not both, not neither)"
        assert (not self.USE_SFC_ATMOS) or self.USE_ATMOSPHERIC, \
            "USE_SFC_ATMOS requires USE_ATMOSPHERIC=True (it appends a surface " \
            "anchor channel to the atmospheric input)"
        assert (not self.USE_SURFACE_PRECIP) or self.USE_SURFACE, \
            "USE_SURFACE_PRECIP requires USE_SURFACE=True (it appends the " \
            "ERA5-Land precipitation channel to the surface input)"
        assert not (not self.SEASONAL_FEATURES and self.SEASONAL_FEATURES_IN_MLP), \
            "SEASONAL_FEATURES_IN_MLP cannot be True if SEASONAL_FEATURES (master switch) is False"


        import torch
        if isinstance(self.DEVICE, str):
            self.DEVICE = torch.device(self.DEVICE)

    def with_in_channels(self, in_channels: int) -> 'Params':
        """Return a new Params instance with IN_CHANNELS set."""
        return dataclasses.replace(self, IN_CHANNELS=in_channels)

    def with_data_paths(self, data_paths) -> 'Params':
        """Return a new Params instance with data paths populated."""
        return dataclasses.replace(
            self,
            ERA5_MAX_TEMP_GLOB=data_paths.ERA5_MAX_TEMP_GLOB,
            ERA5_PRECIP_GLOB=data_paths.ERA5_PRECIP_GLOB,
            ERA5_GEOPOTENTIAL_GLOB=data_paths.ERA5_GEOPOTENTIAL_GLOB,
            ERA5_PRESSURE_LEVEL_DIR=data_paths.ERA5_PRESSURE_LEVEL_DIR,
            ERA5_SURFACE_DIR=data_paths.ERA5_SURFACE_DIR,
            METEO_SWISS_MAX_TEMP_GLOB=data_paths.METEO_SWISS_MAX_TEMP_GLOB,
            METEO_SWISS_PRECIP_GLOB=data_paths.METEO_SWISS_PRECIP_GLOB,
            HI_RES_TOPOGRAPHY_ZARR_PATH=data_paths.HI_RES_TOPOGRAPHY_ZARR_PATH,
        )

    def save_json(self, output_path: Path) -> None:
        """Save this Params instance as JSON to specified path."""
        json_utils.save_dataclass_json(self, output_path)

    @staticmethod
    def load_json(params_json: Path) -> 'Params':
        """Load Params dataclass from JSON file at specified path."""
        return json_utils.load_dataclass_json(Params, params_json)


def is_renku() -> bool:
    return "RENKU_PROJECT_ID" in os.environ


def configure_renku_cuda() -> None:
    """Apply CUDA workarounds for Renku MIG (Multi-Instance GPU) partitions.

    The CUDA caching allocator's "expandable segments" feature requires NVML
    calls that fail on MIG devices. This disables that feature.
    """
    if "RENKU_PROJECT_ID" in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False,max_split_size_mb:512"


def select_device():
    """Select the best available torch device.

    Priority: CUDA > MPS > CPU.
    """
    # Note: we import torch only here, instead of doing it on the
    # top of the module to prevent this being imported before
    # configure_renku_cuda is executed, which would make CUDA
    # usage break.
    import torch

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch, numpy, Python, and CUDA."""
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
