# Config module for parsing inputs from TOML

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def runconfig(config_file_path):
    with open(config_file_path, mode='rb') as fp:
        config = tomllib.load(fp)
    return config
