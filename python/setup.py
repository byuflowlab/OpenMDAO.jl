
from setuptools import setup

setup_args = {'description': 'Create OpenMDAO Components using the Julia programming language',
   'entry_points': {
       'openmdao_component': [
           'juliaexplicitcomp=omjlcomps:JuliaExplicitComp',
           'juliaimplicitcomp=omjlcomps:JuliaImplicitComp'
       ]
    },
   'install_requires': ['openmdao', 'juliapkg', 'juliacall'],
   'keywords': ['openmdao_component'],
   'license': 'MIT',
   'name': 'omjlcomps',
   'packages': ['omjlcomps', 'omjlcomps.test'],
   'version': '0.1.9',
   'include_package_data': True}

setup(**setup_args)
