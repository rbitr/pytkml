import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytkml", # Replace with your own username
    version="0.0.1",
    author="Willows AI",
    author_email="andrew@willows.ai",
    description="Testing for ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rbitr/pytkml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
