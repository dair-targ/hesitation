from setuptools import setup, find_packages

setup(
    name='hesitation',
    version='0.0',
    packages=find_packages(),
    url='https://github.com/dair-targ/hesitation',
    license='',
    author='Vladimir Berkutov',
    author_email='vladimir.berkutov@gmail.com',
    description='',
    install_requires=(
        'matplotlib',
        'numpy',
        'scipy',
    ),
    entry_points=dict(
      console_scripts=[
        'wavsaw-split = wavsaw.split:main',
      ],
    ),
)
