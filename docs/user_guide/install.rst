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

Optional:

- ``skfolio-accelerate[cosmo]`` — native Rust COSMO.rs. See :ref:`cosmo`.
  GitHub ``main`` Python bindings currently expose ``update_q`` /
  ``update_b`` / ``warm_start``. Persistent ``update_p`` / ``update_a`` /
  ``reset`` exist in the Rust solver; build a COSMO.rs checkout that
  exports them (``maturin develop --release --features python``) for the
  persistence experiment. Without those methods the compact path still
  solves, but reconstructs the workspace each fold.

NumPy, SciPy, Clarabel, pandas, and scikit-learn come from skfolio's own
runtime stack.
