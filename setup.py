from setuptools import find_packages
from distutils.core import setup


setup(
    name="get_vECG",
    version="1.0",
    packages=find_packages(),
    long_description="Создает вЭКГ на основе ЭКГ",
    include_package_data=True,
    entry_points={
        "console_scripts": ["get_VECG=main:main"],
    },
    install_requires=[
        'mne',
        'pandas',
        'matplotlib',
        'numpy',
        'scipy',
        'neurokit2',
        'plotly',
        'click'
    ],
)