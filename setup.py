from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    return codecs.open(os.path.join(here, *parts), 'r', encoding='utf8').read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

class Tox(TestCommand):
    user_options = [('tox-args=', 'a', "Arguments to pass to tox")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import tox
        import shlex
        errno = tox.cmdline(args=shlex.split(self.tox_args))
        sys.exit(errno)

setup(
    name='sdeint',
    version=find_version('sdeint', '__init__.py'),
    url='http://github.com/mattja/sdeint/',
    bugtrack_url='https://github.com/mattja/sdeint/issues',
    license='GPLv3+',
    author='Matthew J. Aburn',
    install_requires=['numpy>=1.6'],
    tests_require=['tox', 'scipy>=0.13'],
    cmdclass = {'test': Tox},
    author_email='mattja6@gmail.com',
    description='Numerical integration of stochastic differential equations (SDE)',
    long_description=read('README.rst'),
    packages=['sdeint'],
    platforms='any',
    zip_safe=False,
    keywords = ['stochastic', 'differential equations', 'SDE', 'SODE'],
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        ],
    extras_require={'implicit_algorithms': ['scipy>=0.13']}
)
