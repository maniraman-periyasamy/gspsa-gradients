from setuptools import setup, find_packages

setup(
    name='gspsa_gradients',
    version='1.0',
    packages=find_packages(),
    py_modules=['gspsa_gradients'],
    install_requires=[
        'numpy',
    ],
    extras_require={
        'tfq': ['tensorflow-quantum'],
        'qiskit': ['qiskit'],
    },
)