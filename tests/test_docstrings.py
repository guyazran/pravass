"""The examples in the docstrings run and print what they show."""

import doctest
import importlib
import pkgutil

import pytest

import pravass

MODULES = [pravass] + [
    importlib.import_module(f"pravass.{info.name}") for info in pkgutil.iter_modules(pravass.__path__)
]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_docstring_examples(module):
    result = doctest.testmod(module)
    assert result.failed == 0
