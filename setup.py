from setuptools import find_packages, setup

# minimal setup.py for a src-layout package
setup(
    name="cancer_detection",
    version="0.0.1",
    description="Demo pipeline for cancer vs healthy classification",
    author="Bryan Goggin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    extras_require={"tf": ["tensorflow>=2.10"]},
    python_requires=">=3.8"
)
