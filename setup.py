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
      'scipy==1.10.1',
      'numpy==1.24.4',
      'resampy=0.4.2',
      'panphon==0.20.0',
      'torch==2.0.1',
      'editdistance==0.6.2',
      'torchaudio==2.0.2',
   ]
)
