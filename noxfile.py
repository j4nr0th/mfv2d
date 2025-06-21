"""Config file for NOX."""

import nox


@nox.session
def run_tests(session: nox.Session):
    """Run tests using pytest."""
    session.install(".")
    session.install("pytest", "pytest-cov")
    session.run("pytest", "tests")


@nox.session
def interrogate(session: nox.Session):
    """Check for docstrings."""
    session.install("interrogate")
    session.run("interrogate")


@nox.session
def make_documentation(session: nox.Session):
    """Build the documentation."""
    session.install("-e", ".[docs]")
    session.cd("docs")
    session.run("make", "clean")
    session.run("make", "html")
