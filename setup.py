from setuptools import setup, find_packages

setup(
    name="sensor-fusion-disagreement",
    version="2.0.0",
    description=(
        "Distributed Disagreement-Based Fault Detection and Robust Sensor Fusion "
        "via Weighted Consensus — IEEE Sensors Letters 2026"
    ),
    author="Kushal Khemani and Sujal Kosta",
    author_email="kushal.khemani@gmail.com",
    url="https://github.com/Kushalk0677/Distributed-Disagreement-Based-Fault-Detection-and-Robust-Sensor-Fusion-via-Weighted-Consensus",
    packages=find_packages(exclude=["tests*", "experiments*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "pandas>=1.5",
        "scikit-learn>=1.0",
    ],
    extras_require={"dev": ["pytest>=7.0", "pytest-cov"]},
)
