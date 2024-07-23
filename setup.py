from setuptools import setup

setup_args = {
    "name": "HERA_FRF_cov",
    "author": "M. Wilensky",
    "url": "https://github.com/mwilensky768/HERA_FRF_cov",
    "license": "BSD",
    "description": "Calculate noise covariances after HERA fringe rate filtering",
    "package_dir": {"HERA_FRF_cov": "HERA_FRF_cov"},
    "packages": ["HERA_FRF_cov"],
    "install_requires": ["numpy", "hera-calibration", "hera-filters", "astropy",
                         "pyyaml"],
    "zip_safe": False
}

if __name__ == '__main__':
    setup(**setup_args)