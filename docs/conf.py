"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mfv2d"
copyright = "2025, Jan Roth"  # noqa: A001
author = "Jan Roth"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "jupyter_sphinx",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritence": True,
}
autodoc_type_aliases = {
    "npt.ArrayLike": "array_like",
    "npt.NDArray": "array",
    "npt.NDArray[np.float64]": "array[float64]",
    "npt.NDArray[np.uint32]": "array[uint32]",
    "npt.NDArray[np.uint64]": "array[uint64]",
    "npt.NDArray[np.double]": "array[double]",
    "npt.NDArray[np.floating]": "array[floating]",
}


# -- Options for BibTeX extension ---------------------------------------------
bibtex_bibfiles = ["refs.bib"]

# -- Options for Intersphinx -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pyvista": ("https://docs.pyvista.org/", None),
}


# -- Options for Napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True


# -- Options for Sphinx Gallery ----------------------------------------------
# https://sphinx-gallery.github.io/stable/index.html
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "subsection_order": [
        "../examples/steady",
        "../examples/unsteady",
        "../examples/refinement",
    ],
    "gallery_dirs": "auto_examples",
    "reference_url": {
        # The module you locally document uses None
        "mfv2d": None,
    },
    "image_scrapers": ("matplotlib", "pyvista"),
}
import pyvista  # noqa: E402

pyvista.BUILDING_GALLERY = True
