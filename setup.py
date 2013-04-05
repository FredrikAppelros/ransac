from setuptools import setup

setup(name="ransac",
      version='0.1',
      description='Robust method for fitting a model to observed data.',
      author='Fredrik Appelros, Carl Ekerot',
      author_email='fredrik.appelros@gmail.com, kalle@implode.se',
      url='https://github.com/FredrikAppelros/ransac',
      py_modules=['ransac'],
      install_requires=['numpy'],
)

