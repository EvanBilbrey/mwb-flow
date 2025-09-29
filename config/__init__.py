from pathlib import Path
import tomli

path = Path(__file__).parent / 'mwb_settings.toml'
with path.open(mode='rb') as f:
    settings = tomli.load(f)

GRIDMET_PARAMS = settings['GRIDMET']['GRIDMET_PARAMS']
GRIDMET_BOUNDS = settings['GRIDMET']['GRIDMET_BOUNDS']
GRIDMET_NROWS = settings['GRIDMET']['GRIDMET_NROWS']
GRIDMET_NCOLS = settings['GRIDMET']['GRIDMET_NCOLS']
GRIDMET_XRES = settings['GRIDMET']['GRIDMET_XRES']
GRIDMET_YRES = settings['GRIDMET']['GRIDMET_YRES']
INPUT_VARS = settings['DLEM']['INPUT_VARS']
DSET_COORDS = settings['DLEM']['DSET_COORDS']
