from setuptools import find_packages, setup
from typing import List

HYPEN_DOT_E = "-e ."

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns a list of requirements from the given file.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n","") for req in requirements]  

        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)

    return requirements

# ------------------- Setup function -------------------
setup(
    name="ml_project",
    version="0.0.1",
    author="Ram",
    author_email="rammahto645@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
