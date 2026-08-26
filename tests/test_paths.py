from pathlib import Path

from wnba_edges.paths import repository_root


def test_repository_root_ignores_unrelated_python_data_directory(tmp_path):
    python_root = tmp_path / "python" / "lib" / "python3.12"
    module = python_root / "site-packages" / "wnba_edges" / "cli.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (python_root / "data").mkdir()

    checkout = tmp_path / "checkout"
    (checkout / "data").mkdir(parents=True)
    (checkout / "pyproject.toml").touch()

    assert repository_root(module, cwd=checkout) == checkout.resolve()


def test_repository_root_uses_source_checkout(tmp_path):
    checkout = tmp_path / "repo"
    module = checkout / "src" / "wnba_edges" / "cli.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (checkout / "data").mkdir()
    (checkout / "pyproject.toml").touch()

    assert repository_root(module, cwd=Path("C:/unrelated")) == checkout.resolve()
