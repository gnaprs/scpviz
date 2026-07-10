# functions to add here
# check for installed packages?

# From HRTAtlas (https://housekeeping.unicamp.br/?homePageGlobal), Hounkpe et al. 2020 https://doi.org/10.1093/nar/gkaa609
# Housekeeping genes for human and mouse in __assets__/MostStable_Human.csv and __assets__/MostStable_Mouse.csv
# These are the most stable genes in the human and mouse genome, respectively, and are used for normalization in scpviz

import os
import pandas as pd

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths to the assets
human_file_path = os.path.join(current_dir, '__assets__', 'MostStable_Human.csv')
mouse_file_path = os.path.join(current_dir, '__assets__', 'MostStable_Mouse.csv')

# BUG: assets does not install with package, so these won't run
# Read the CSV files
# housekeeping_HUMAN = pd.read_csv(human_file_path)
# housekeeping_MOUSE = pd.read_csv(mouse_file_path)

# set function to get date and time
import datetime
def get_datetime():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Keep in sync with [project] dependencies and [project.optional-dependencies] in pyproject.toml
def print_versions():
    from importlib.metadata import PackageNotFoundError, version

    # [project] dependencies
    core = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "upsetplot",
        "scikit-learn",
        "scipy",
        "umap-learn",
        "adjustText",
        "anndata",
        "requests",
        "matplotlib_venn",
        "pyarrow",
        "scanpy",
        "gseapy",
        "openpyxl",
        "biopython",
    ]
    # optional-dependencies.sc
    opt_sc = [
        "directlfq",
        "pimms-learn",
        "harmonypy",
        "leidenalg",
        "igraph",
        "scikit-misc",
    ]
    # optional-dependencies.notebook
    opt_notebook = ["IPython"]
    # optional-dependencies.dev
    opt_dev = ["pytest", "coverage", "flake8", "pytest-cov"]

    def _print_group(title: str, packages: list[str]) -> None:
        print(title)
        for package in packages:
            try:
                print(f"  {package}", version(package))
            except PackageNotFoundError:
                print(f"  {package}", "not installed")

    try:
        scpviz_ver = version("scpviz")
    except PackageNotFoundError:
        scpviz_ver = "unknown"
    print(f"scpviz version: {scpviz_ver}")
    
    print("Date and time: ", get_datetime())
    _print_group("Core dependencies:", core)
    _print_group("Optional [sc]:", opt_sc)
    _print_group("Optional [notebook]:", opt_notebook)
    _print_group("Optional [dev]:", opt_dev)

GLOBAL_DEBUG = False

# look into session info
# from session_info2 import session_info
# sinfo = session_info(os=True, cpu=True, gpu=True, dependencies=True)
# if file is not None:
#     print(sinfo, file=file)
#     return
# return sinfo