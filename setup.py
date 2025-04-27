import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='heurisp',
    version='0.3.0',
    packages=setuptools.find_packages(),
    description='Heuristics for Sampling-based Path Planner',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gabel Liemann',
    author_email='troubleli233@gmail.com',
    url='https://github.com/liespace/pyHSP',
    license='MIT',
    install_requires=[
        'numpy>=1.26,<2',
        'numba>=0.57,<0.61',
        'bottleneck>=1.3.8 ',
        'rsplan>=1.0',
        'matplotlib',
        'opencv-python',
        'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
)
