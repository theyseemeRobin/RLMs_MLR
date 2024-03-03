from setuptools import setup

with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='RLMs Machine Learning Recipe',
    description="A simple template for machine learning projects",
    install_requires=requirements,
)
