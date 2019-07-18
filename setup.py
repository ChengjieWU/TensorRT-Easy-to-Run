import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TRE",
    version="0.0.1",
    author="Chengjie Wu",
    author_email="wcj509@163.com",
    description="TensorRT Running Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChengjieWU/TRE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux"
    ],
)
