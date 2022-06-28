from setuptools import setup, find_packages

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Education",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "Operating System :: Unix",
    "Operating System :: MacOS :: MacOS X",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

setup(
    name="basic_rfregressor",
    version="0.0.1",
    description="A very basic regression model",
    url="",
    author="saigodha",
    author_email="saikumar.godha@tigeranalytics.com",
    license="MIT",
    classifiers=classifiers,
    keywords="brfregressor",
    packages=find_packages(),
    install_requires=[""],
)
