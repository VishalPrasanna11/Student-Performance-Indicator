from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path) -> List[str]:
    # This functions returns the requirements from the file
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(

    name='student_performance_prediction',
    version='0.1',
    author='Vishal Prasanna',
    author_email="prasanna.vi@northeastern.edu",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='Predicting student performance using machine learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)