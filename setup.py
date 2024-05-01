import setuptools
import platform
import subprocess

import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
# from create_pycharm_run_configurations import create_run_configurations
# create_run_configurations(os.path.join(current_dir, 'PycharmRunConfigurations'))
sys.path.remove(current_dir)

# Function to read static requirements from requirements.txt
def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as file:
        return file.read().splitlines()


def check_qt_installation():
    """ Check if Qt is properly installed and can be configured """
    try:
        # Attempt to query Qt version using qmake or other relevant commands
        result = subprocess.run(['qmake', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Qt installation found. Version:", result.stdout.decode())
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Qt installation check failed:", e)
        return False
# Static dependencies
static_requirements = load_requirements()


dynamic_requirements = []

# Dynamic dependencies based on Ubuntu version and Qt installation status
if check_qt_installation():
    version = platform.version()
    if 'Ubuntu' in version and '18.04' in version:
        dynamic_requirements.append('PyQt5')
        print('SI_Toolkit is designed to work with PyQt6 which does not support Ubuntu 18.04.\n'
              'You can find however on Github a separate branch with PyQt5 which will be installed now for you.')
    else:
        dynamic_requirements.append('PyQt6')
else:
    print("Qt is not properly installed on this system, skipping PyQt installation.")

# Combine static and dynamic dependencies
requirements = static_requirements + dynamic_requirements

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SI_Toolkit",
    version="0.0.1",
    author="Sensors Group",
    author_email="marcin.p.paluch@gmail.com",
    description="Set of scripts for system identification with neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SensorsINI/SI_Toolkit",
    project_urls={
        "SI_Toolkit": "https://github.com/SensorsINI/SI_Toolkit",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src'),
    python_requires=">=3.8",
    install_requires=requirements
)