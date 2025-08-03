from setuptools import setup, find_packages


def parse_txt(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name='rlm',
    version='0.1.0',
    package_dir={'': '.'},
    packages=find_packages(),
    python_requires='>=3.11',
    # install_requires=parse_txt('requirements.txt')
    author='slahan',
    description='basic implementations of important rl algorithms',
    url='https://github.com/sujeet1024/modern_rl'
)