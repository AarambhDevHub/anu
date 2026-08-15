def test_scaffold_imports() -> None:
    import data  # noqa: F401
    import model  # noqa: F401


def test_python_version() -> None:
    import sys

    assert sys.version_info >= (3, 12)
