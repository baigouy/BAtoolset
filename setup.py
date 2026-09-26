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
    packages=setuptools.find_namespace_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        "Markdown",
        "matplotlib>=3.5.2",
        "numpy>=2.0.0",
        "Pillow>=8.1.2",
        "read-lif",
        "scikit-image>=0.19.3",
        "scipy>=1.7.3",
        "scikit-learn>=1.0.2",
        "tifffile>=2021.11.2",
        "tqdm",
        "natsort",
        "numexpr",
        "urllib3",
        "pandas",
        "elasticdeform",
        "roifile",
        "prettytable",
        "pyperclip",
        "Deprecated",
        "Requests",
        "pyautogui",
        "imagecodecs",
        "psutil",
        "zarr",
    ],
    python_requires='>=3.7, <3.12' # from 04/05/23 colab is using python 3.10.11
)