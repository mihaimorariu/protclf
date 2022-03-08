import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "protclf"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "m2r2",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

autoclass_content = "both"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "requirements.txt"]
html_theme = "sphinx_rtd_theme"
