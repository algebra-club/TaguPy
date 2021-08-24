"""Top-level package for TaguPy."""

from . import manager
from . import simulator
from . import utils
from .core import analyzer, generator, reporter

__all__ = [
    'analyzer',
    'generator',
    'reporter',
    'manager',
    'simulator',
    'utils'
]

__author__ = """Sei Takeda"""
__email__ = 'sei06k14@gmail.com'
__version__ = '0.1.0'
