from setuptools import setup, find_packages

setup(
    name='finidiff',
    version='0.0.1',
    packages=find_packages(),
    package_data={},

    install_requires=['numpy', 'scipy'],

    author='G. D. McBain',
    author_email='gmcbain@protonmail.com'

)
