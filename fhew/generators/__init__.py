"""Hop-generator library.

Importing this package registers every generator family in the name-keyed
registry (paper Section 5). The journal adds chaotic / learned adversaries by
dropping a module here and decorating the class with ``@register(name)``.
"""

from fhew.generators.base import (
    Generator,
    get_generator_cls,
    iter_registry,
    make_generator,
    register,
    registered_names,
)

# Importing the modules triggers their @register side effects.
from fhew.generators import lfsr as _lfsr  # noqa: F401,E402
from fhew.generators import lcg as _lcg  # noqa: F401,E402
from fhew.generators import markov as _markov  # noqa: F401,E402
from fhew.generators import csprng as _csprng  # noqa: F401,E402

from fhew.generators.lfsr import FilteredLFSRGenerator, LFSRGenerator  # noqa: E402
from fhew.generators.lcg import LCGGenerator, WeakSeedGenerator  # noqa: E402
from fhew.generators.markov import MarkovGenerator  # noqa: E402
from fhew.generators.csprng import CSPRNGGenerator  # noqa: E402

__all__ = [
    "Generator",
    "register",
    "make_generator",
    "get_generator_cls",
    "registered_names",
    "iter_registry",
    "LFSRGenerator",
    "FilteredLFSRGenerator",
    "LCGGenerator",
    "WeakSeedGenerator",
    "MarkovGenerator",
    "CSPRNGGenerator",
]
