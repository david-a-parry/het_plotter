try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import re
v_file = "het_plotter/version.py"
v_line = open(v_file, "rt").read()
v_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
match = re.search(v_re, v_line, re.M)
if match:
    verstr = match.group(1)
else:
    raise RuntimeError("Unable to find version string in {}.".format(v_file))

setup(
    name="het_plotter",
    packages=["het_plotter"],
    version=verstr,
    description="Find overlapping CNVs",
    author="David A. Parry",
    author_email="david.parry@igmm.ed.ac.uk",
    url='https://git.ecdf.ed.ac.uk/dparry/het_plotter',
    license='MIT',
    install_requires=['pysam', 'numpy', 'pandas', 'seaborn'],
    scripts=["bin/plot_heterozygosity"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        ],
)
