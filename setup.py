from setuptools import setup, find_packages

setup(
    name='gspsa-gradients',
    version='0.1.0',
    packages=find_packages(),
    py_modules=['gspsa-gradients'],
    install_requires=[
        'numpy',
    ],
    extras_require={
        'tfq': ['tensorflow-quantum', 'tensorflow'],
        'qiskit': ['qiskit', 'qiskit-algorithms'],
    },
)