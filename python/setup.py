from distutils.core import setup

setup(name='omjl',
      version='0.0.1',
      description='OpenMDAO for Julia',
      author='Andrew Ning',
      packages=['omjl'],
      install_requires=[
          'openmdao>=2.4.0',
          'numpy>=1.14.1',
      ],
      zip_safe=False)
