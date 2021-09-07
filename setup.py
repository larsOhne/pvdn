from setuptools import setup, find_packages
import os

lib_dir = os.path.dirname(os.path.realpath(__file__))
requirements_path = lib_dir + "/requirements.txt"
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='pvdn',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/larsOhne/pvdn',
    license='Creative Commons Legal Code ',
    author='Lars Ohnemus, Lukas Ewecker, Ebubekir Asan, Stefan Roos, Simon Isele, Jakob Ketterer, Leopold Müller, and Sascha Saralajew',
    author_email='',
    description='Tools for working with the PVDN dataset.',
    install_requires=install_requires
)
