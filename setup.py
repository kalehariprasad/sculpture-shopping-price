
from setuptools import setup,find_packages
from typing import List 

REQUIREMENTS_FILE_NAME='requirements.txt'
HYPEN_E_DOT='-e .'

def get_requirements_list()->List[str]:
    with open(REQUIREMENTS_FILE_NAME) as requirement_file:
        requiremet_list=requirement_file.readlines()
        requiremet_list=[ requirement_name.replace ('\n','')for requirement_name in requiremet_list ]
        
        if HYPEN_E_DOT in requiremet_list:
            requiremet_list.remove(HYPEN_E_DOT)
        return requiremet_list



setup(name='Sculpture price',
      version='1.0',
      description='Sculpture price prediction system',
      author='Hari Prasad',
      author_email='hariprasad9693@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements_list()
     )