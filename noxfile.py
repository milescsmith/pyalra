import os

import nox

package = "pyalra"
# nox.options.sessions = ["lint", "black", "tests"]
nox.options.sessions = ["tests"]
locations = "src", "tests", "noxfile.py", "docs/conf.py"
os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})


@nox.session(python="3.10")
def tests(session: nox.session) -> None:
    # args = session.posargs or locations
    # install_with_constraints(session, ".")
    # install_with_constraints(session, "pytest")
    session.run_always("pdm", "install", "-G", "test", external=True)
    session.run("pytest")


@nox.session()
def black(session: nox.session) -> None:
    """Run black code formatter."""
    session.install("black")
    session.run_always("pdm", "install", "-G", "test", external=True)
    session.run("black", "src")


@nox.session
def lint(session: nox.session) -> None:
    """Lint using ruff."""
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", *args)
