"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

# -- Path setup ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# -- Project information -------------------------------------------------------
project = "skfolio-accelerate"
copyright = f"2024-{datetime.now().year}, skfolio-accelerate contributors"
author = "skfolio-accelerate contributors"
try:
    version = pkg_version("skfolio-accelerate")
except PackageNotFoundError:
    version = "0.1.0"
release = version

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
default_role = "literal"
add_function_parentheses = False
# FileNameSortKey (and similar callables) in sphinx_gallery_conf are not
# pickleable; ignore the resulting environment-cache warning under -W.
suppress_warnings = ["config.cache"]

# -- Autodoc / autosummary / numpydoc ------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_use_plots = True

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "skfolio": ("https://skfolio.org", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}

# -- HTML theme ----------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static", "figures"]
html_title = "skfolio-accelerate"
html_short_title = "skfolio-accelerate"
html_baseurl = "https://carlonicolini.github.io/skfolio-accelerate/"

html_theme_options = {
    "github_url": "https://github.com/CarloNicolini/skfolio-accelerate",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "logo": {
        "text": "skfolio-accelerate",
    },
    "icon_links": [],
}

html_context = {
    "default_mode": "light",
}

# -- Plotly (Sphinx-Gallery) ---------------------------------------------------
# Capture interactive Plotly figures in gallery examples. kaleido provides PNG
# thumbnails via plotly_sg_scraper; without it the HTML repr is still embedded.
try:
    import plotly.io as pio
    from plotly.io._sg_scraper import plotly_sg_scraper
    from sphinx_gallery.sorting import FileNameSortKey

    pio.renderers.default = "sphinx_gallery_png"
    _image_scrapers = ("matplotlib", plotly_sg_scraper)
    _subsection_order = FileNameSortKey
except ImportError:  # pragma: no cover - docs extra missing plotly
    _image_scrapers = ("matplotlib",)
    _subsection_order = None

# -- Sphinx-Gallery ------------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": ["../examples/getting_started"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"massive_path_predict\.py",
    "download_all_examples": False,
    "plot_gallery": True,
    "remove_config_comments": True,
    "doc_module": ("skfolio_accelerate",),
    "reference_url": {"skfolio_accelerate": None},
    "backreferences_dir": "generated/backreferences",
    "image_scrapers": _image_scrapers,
}
if _subsection_order is not None:
    sphinx_gallery_conf["within_subsection_order"] = _subsection_order

# Local/optional fast builds can skip executing examples. Continuous
# integration always executes the gallery (including Plotly speedup figures).
if os.environ.get("SKFOLIO_ACCELERATE_DOCS_FAST") == "1":
    sphinx_gallery_conf["plot_gallery"] = False

# -- Copybutton ----------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
