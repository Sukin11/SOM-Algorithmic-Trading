from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="som_algo_trading",
    version="1.0.0",
    description="SOM-based algorithmic trading pipeline for market regime clustering and T+1 forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SOM Algo Trading",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "numpy>=1.23",
        "scikit-learn>=1.2",
        "minisom>=2.3",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5"],
        "dev": ["pytest>=7.0", "pytest-cov", "matplotlib>=3.5"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
