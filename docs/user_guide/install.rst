.. _install:

************
Installation
************

Install using pip
*****************

`skfolio-accelerate` targets skfolio 1.x. Install from PyPI (when published) or
from a local checkout:

.. code:: console

    $ python -m venv .venv
    $ source .venv/bin/activate
    $ pip install skfolio-accelerate

For development:

.. code:: console

    $ pip install -e ".[dev]"
    $ pytest

Documentation dependencies (Sphinx, gallery, theme):

.. code:: console

    $ pip install -e ".[docs]"
    $ cd docs && make html

Dependencies
************

Runtime:

- python (>= 3.10)
- skfolio (>= 1.0, < 2)
- osqp (>= 1.0, < 2)

NumPy, SciPy, Clarabel, pandas, and scikit-learn come from skfolio's own
runtime stack.

Optional COSMO extra
********************

``MeanRisk(solver="COSMO")`` uses COSMO.jl as a compact ADMM engine. This is
opt-in; the default compact path remains OSQP (variance) and Clarabel
(scenario risks). Install Julia, then:

.. code:: console

    $ pip install -e ".[cosmo]"
    $ julia -e 'using Pkg; Pkg.add("COSMO")'

``juliacall`` starts a process-local Julia runtime on first use and installs
``COSMO.jl`` into that environment (declared in ``juliapkg.json``). Missing
COSMO does not affect ``import skfolio_accelerate`` or the default backends.
