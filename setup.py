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
    # only batoolset: a stale build/lib (pre-split GUI modules) was packaged too
    packages=setuptools.find_namespace_packages(include=['batoolset', 'batoolset.*']),
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
        # only img.elastic_deform needs it; no wheels for 3.13+ (pip install batoolset[deform] builds it)
        "elasticdeform; python_version<'3.13'",
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
    extras_require={
        'deform': ["elasticdeform"],
    },
    python_requires='>=3.10',  # numpy>=2 needs 3.9+, CI tests 3.10+; no upper cap
)