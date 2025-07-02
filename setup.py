# python geopack  setup.py
import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='geopack-vectorized',
    version='1.1.1',
    author='Sheng Tian',
    author_email='ts0110@atmos.ucla.edu',
    description='Python implementation of geopack and Tsyganenko models with optimized vectorization',
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
