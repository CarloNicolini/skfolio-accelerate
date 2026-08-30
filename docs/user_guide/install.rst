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

- ``skfolio-accelerate[cosmo]`` — native Rust COSMO.rs from
  https://github.com/CarloNicolini/COSMO.rs (``update_q`` / ``update_b`` /
  ``update_p`` / ``update_a`` / ``reset`` / ``warm_start``). See :ref:`cosmo`.
  ``update_q`` / ``update_b`` do not refactor. Same-sparsity ``update_p``
  numerically refactors; ``update_a`` and a changed ``P`` pattern rebuild
  the KKT system on the next ``solve``. Without those methods (older
  wheels) the compact path still solves, but reconstructs the workspace
  each fold.

NumPy, SciPy, Clarabel, pandas, and scikit-learn come from skfolio's own
runtime stack.
