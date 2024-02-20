from setuptools import setup

from pathlib import Path
from io import open

with open(Path(__file__).parent / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup_args = {
   'description': 'Create OpenMDAO Components using the Julia programming language',
   'long_description': long_description,
   'long_description_content_type': 'text/markdown',
   'entry_points': {
       'openmdao_component': [
           'juliaexplicitcomp=omjlcomps:JuliaExplicitComp',
           'juliaimplicitcomp=omjlcomps:JuliaImplicitComp'
       ]
    },
   'install_requires': ['openmdao~=3.26', 'juliapkg~=0.1.10', 'juliacall~=0.9.13'],
   'keywords': ['openmdao_component'],
   'license': 'MIT',
   'name': 'omjlcomps',
   'packages': ['omjlcomps', 'omjlcomps.test'],
   'version': '0.2.4',
   'include_package_data': True}

setup(**setup_args)
