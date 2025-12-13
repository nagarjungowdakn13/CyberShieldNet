from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cybershieldnet",
    version="1.0.0",
    author="Nagarjun Gowda K N",
    author_email="nagarjun@gmail.com",
    description="A novel multi-modal fusion framework for predictive cyber threat intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nagarjungowda/CyberShieldNet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cybershieldnet-train=scripts.train_model:main",
            "cybershieldnet-evaluate=scripts.evaluate_model:main",
            "cybershieldnet-serve=scripts.deploy_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cybershieldnet": ["config/*.yaml", "data/sample/*.json"],
    },
)