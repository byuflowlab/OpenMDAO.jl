
from setuptools import setup

setup_args = {'description': 'Create OpenMDAO Components using the Julia programming language',
   'entry_points': {
       'openmdao_component': [
           'juliaexplicitcomp=omjlcomps:JuliaExplicitComp',
           'juliaimplicitcomp=omjlcomps:JuliaImplicitComp'
       ]
    },
   'install_requires': ['openmdao~=3.26.0', 'juliapkg~=0.1.10', 'juliacall~=0.9.13'],
   'keywords': ['openmdao_component'],
   'license': 'MIT',
   'name': 'omjlcomps',
   'packages': ['omjlcomps', 'omjlcomps.test'],
   'version': '0.2.1',
   'include_package_data': True}

setup(**setup_args)
