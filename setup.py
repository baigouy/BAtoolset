import setuptools
from batoolset.version import __VERSION__,__EMAIL__,__AUTHOR__,__DESCRIPTION__, __URL__

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='batoolset', # shall I do it like that or not --> try on testpy first
    version=__VERSION__,
    author=__AUTHOR__,
    author_email=__EMAIL__,
    description=__DESCRIPTION__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=__URL__,
    package_data={'': ['*.md','*.json']},
    license='BSD',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        "czifile",
        "Markdown",
        "matplotlib>=3.5.2",
        "numpy<1.26.4", # <1.26.4 for compat with javabridge
        "Pillow>=8.1.2",
        "PyQt6",
        "read-lif",
        "scikit-image>=0.19.3",
        "scipy>=1.7.3",
        "scikit-learn>=1.0.2",
        "tifffile>=2021.11.2",
        "tqdm",
        "natsort",
        "numexpr",
        "urllib3",
        "qtawesome",
        "pandas",
        "numba",
        "elasticdeform",
        "roifile",
        "prettytable",
        "pyperclip",
        "QtPy>=2.1.0",
        "Deprecated",
        "Requests",
        # "python-bioformats",
        # "python-javabridge",
        "pyautogui",
        "imagecodecs",
        "psutil",
        # "zarr",
    ],
    extras_require={'all': [
        "python-javabridge",
        "python-bioformats",
    ],
    },
    python_requires='>=3.7, <3.11' # from 04/05/23 colab is using python 3.10.11
)