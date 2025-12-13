from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read description (for PyPI/GitHub)
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="efficient-diffusion-loader",
    version="0.1.0",
    author="Yash Sinha",
    author_email="your.email@example.com",
    description="Optimize Stable Diffusion inference on consumer hardware with Fractional Batching.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/efficient-diffusion-loader",
    
    # This automatically finds your code inside 'src'
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    install_requires=requirements,
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)