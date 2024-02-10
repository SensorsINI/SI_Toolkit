import setuptools
import platform


# Function to read static requirements from requirements.txt
def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as file:
        return file.read().splitlines()


# Static dependencies
static_requirements = load_requirements()


dynamic_requirements = []
# Dynamic dependencies based on Ubuntu version
version = platform.version()
if 'Ubuntu' in version and '18.04' in version:
    dynamic_requirements.append('PyQt5')
    print('SI_Toolkit is designed to work with PyQt6 which does not support Ubuntu 18.04.\n'
          'You can find however on Github a separate branch with PyQt5 which will be installed now for you.')
else:
    dynamic_requirements.append('PyQt6')

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
