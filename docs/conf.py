# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

# -- Project information -----------------------------------------------------

try:
    info = metadata("scxpand")
    project_name = info["Name"]
    author = info.get("Author") or "The scXpand Team"
    release = info["Version"]
except PackageNotFoundError:
    project_name = "scXpand"
    author = "The scXpand Team"
    # Get version from package
    try:
        from scxpand import __version__

        release = __version__
    except ImportError:
        release = "0.1.0"

project = project_name
version = release
project_copyright = f"{datetime.now():%Y}, {author}"
bibtex_bibfiles = ["references.bib"]
# -- HTML context for GitHub integration ------------------------------------
repository_url = f"https://github.com/yizhak-lab-ccg/{project_name}"

html_context = {
    "display_github": True,
    "github_user": "yizhak-lab-ccg",
    "github_repo": project_name,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
if os.environ.get("READTHEDOCS") == "True":
    html_context["READTHEDOCS"] = True

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
]

autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_custom_sections = [("Params", "Parameters")]

# Configure autodoc to highlight parameter names
autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Enable parameter highlighting
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ModelType": ":class:`scxpand.util.classes.ModelType`",
    "AnnData": ":class:`anndata.AnnData`",
    "Path": ":class:`pathlib.Path`",
    "np.ndarray": ":class:`numpy.ndarray`",
    "torch.Tensor": ":class:`torch.Tensor`",
}

myst_heading_anchors = 6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")

# Additional myst-nb configuration
nb_execution_show_tb = True
nb_execution_raise_on_error = False

nb_output_stderr = "remove"
nb_execution_mode = "off"  # Use existing outputs, don't execute notebooks
nb_merge_streams = True
nb_execution_timeout = 60

# Suppress warnings
suppress_warnings = [
    "mystnb.unknown_mime_type",
    "autosummary.import_object",
    "duplicate_object",
    "ref.doc",
]

# ReadTheDocs-specific notebook settings
if os.environ.get("READTHEDOCS") == "True":
    # On ReadTheDocs, use 'off' mode to avoid execution issues
    nb_execution_mode = "off"
    # Increase timeout for complex notebooks
    nb_execution_timeout = 120
    # Disable execution for ReadTheDocs
    nb_execution_allow_errors = True
    nb_execution_excludepatterns = []
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "scanpy": ("https://scanpy.readthedocs.io/en/stable", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "cycler": ("https://matplotlib.org/cycler", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable", None),
    "leidenalg": ("https://leidenalg.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "seaborn": ("https://seaborn.pydata.org", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "lightgbm": ("https://lightgbm.readthedocs.io/en/latest", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
    "structlog": ("https://www.structlog.org/en/stable", None),
    "shap": ("https://shap.readthedocs.io/en/latest", None),
    "bbknn": ("https://bbknn.readthedocs.io/en/latest", None),
    "scrublet": ("https://scrublet.readthedocs.io/en/stable", None),
    "scirpy": ("https://scirpy.readthedocs.io/en/stable", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# Include custom CSS
html_css_files = [
    "custom.css",
]

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
    "logo": {
        "image_light": "https://raw.githubusercontent.com/yizhak-lab-ccg/scXpand/main/docs/images/scXpand_logo_gray.png",
        "image_dark": "https://raw.githubusercontent.com/yizhak-lab-ccg/scXpand/main/docs/images/scXpand_logo_gray.png",
    },
    "show_navbar_depth": 1,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}
