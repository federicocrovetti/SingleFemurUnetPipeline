import os

try:
    from setuptools import setup
    from setuptools import find_packages

except ImportError:
  from distutils.core import setup
  from distutils.core import find_packages

def get_requires (requirements_filename):
  '''
  What packages are required for this module to be executed?
  Parameters
  ----------
    requirements_filename : str
      filename of requirements (e.g requirements.txt)
  Returns
  -------
    requirements : list
      list of required packages
  '''
  with open(requirements_filename, 'r') as fp:
    requirements = fp.read()

  return list(filter(lambda x: x != '', requirements.split()))



def read_description (readme_filename):
  '''
  Description package from filename
  Parameters
  ----------
    readme_filename : str
      filename with readme information (e.g README.md)
  Returns
  -------
    description : str
      str with description
  '''

  try:

    with open(readme_filename, 'r') as fp:
      description = '\n'
      description += fp.read()

    return description

  except IOError:
    return ''

here = os.path.abspath(os.path.dirname(__file__))

#Package-Metadata
NAME = "UnetSingleFemurSegmentation"
DESCRIPTION = 'Package for Semantic Segmentation of CT images of femurs with a single label'
URL = 'https://github.com/federicocrovetti/2DUnetFemurSegmentation'
EMAIL = 'federico.crovetti@studio.unibo.it'
AUTHOR = 'Federico Crovetti, Riccardo Biondi'
VERSION = '1.0.0'
KEYWORDS = 'artificial-intelligence machine-learning deep-learning medical-imaging tensorflow u-net single-femur femur'
REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
README_FILENAME = os.path.join(here, 'README.md')
try:
  LONG_DESCRIPTION = read_description(README_FILENAME)

except IOError:
  LONG_DESCRIPTION = DESCRIPTION




setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    mantainer=AUTHOR,
    mantainer_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=URL,
    keywords=KEYWORDS,
    packages=find_packages(include=['SFUNet', 'SFUNet.*'], exclude=('tests')),
    include_package_data=True, # no absolute paths are allowed
    platforms='any',
    install_requires=get_requires(REQUIREMENTS_FILENAME),

    classifiers=[
        "Programming Language :: Python :: 3",
        #Operating System :: POSIX :: Linux
        Operating System :: Microsoft :: Windows :: Windows 10
        Operating System :: Microsoft :: Windows :: Windows 11
        Environment :: MacOS X :: Aqua
        Environment :: MacOS X :: Carbon
        Environment :: MacOS X :: Cocoa
    ],
    python_requires='>=3.7.11',
    license = 'MIT'
)
