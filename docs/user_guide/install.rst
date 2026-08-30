************
Installation
************

.. warning::

   **Experimental library.** APIs and numerical paths may change between
   releases. Pin a version and re-check results against native skfolio after
   upgrades.

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
- highspy (>= 1.8)

NumPy, SciPy, Clarabel, pandas, and scikit-learn come from skfolio's own
runtime stack.

Optional COSMO.rs backend
*************************

The experimental persistent COSMO path needs a native build of
`COSMO.rs <https://github.com/CarloNicolini/COSMO.rs>`_::

    $ pip install -e ".[cosmo]"
    # or: clone COSMO.rs and `maturin develop --release --features python`

Then pass ``backend="cosmo"`` or ``MeanRisk(solver="COSMO")``. See
:ref:`cosmo`.
