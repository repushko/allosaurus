from setuptools import setup,find_packages

setup(
   name='allosaurus',
   version='1.0.2',
   description='a multilingual phone recognizer',
   author='Xinjian Li',
   author_email='xinjianl@cs.cmu.edu',
   url="https://github.com/xinjli/allosaurus",
   packages=find_packages(),
   include_package_data=True,
   install_requires=[
      'scipy',
      'numpy',
      'resampy',
      'panphon',
      'torch==1.8.1',
      'editdistance',
      'torchaudio==0.9.0',
   ]
)
