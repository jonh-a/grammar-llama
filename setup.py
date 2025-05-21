from setuptools import setup

setup(
    name = 'grammar_llama',
    version = '0.0.1',
    packages = ['grammar_llama'],
    entry_points = {
        'console_scripts': [
            "grammar_llama = grammar_llama.main:main"
        ]
    }
)