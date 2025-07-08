# python geopack  setup.py
import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='geopack-vectorized',
    version='1.1.3',
    author='geopack-vectorize contributors',
    author_email='',
    description='Vectorized Python implementation of geopack and Tsyganenko models (fork of tsssss/geopack)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url= 'https://github.com/Butadiene/geopack-vectorize',
    install_requires= ['numpy','scipy'],
    platforms= ['any'],
    license= 'MIT',
    keywords= ['geopack','space physics','Tsyganenko model'],
    packages= setuptools.find_packages(),
    package_data={'':['*.txt','*.md']},
    classifiers= [
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Physics'
    ],
)
