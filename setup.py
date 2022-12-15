from setuptools import setup, find_packages

setup(
    name='gym_anytrading_stocks',
    version='0.1',
    packages=find_packages(),

    author='Cesar Bonadio',
    author_email='cesar.bonadio@gmail.com',
    license='MIT',

    install_requires=[
        'gym>=0.12.5',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'matplotlib>=3.1.1'
    ],

)