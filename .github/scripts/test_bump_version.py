import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bump_version", Path(__file__).with_name("bump_version.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for path, text in {
            "Cargo.toml": '[package]\nversion = "0.11.0"\n',
            "python/Cargo.toml": '[package]\nversion = "0.11.0"\n',
            "pyproject.toml": '[project]\nversion = "0.11.0"\n',
            "python/src/lib.rs": 'm.add("__version__", "0.11.0")?;\n',
            "changelog.md": "## Unreleased\n\n- Fix\n\n## 0.11.0\n\n- Previous\n",
        }.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def snapshot(self):
        return {
            str(p.relative_to(self.root)): p.read_text()
            for p in self.root.rglob("*")
            if p.is_file()
        }

    def test_components(self):
        original = self.snapshot()
        for bump, expected in [("patch", "0.11.1"), ("minor", "0.12.0"), ("major", "1.0.0")]:
            with self.subTest(bump=bump):
                for path, text in original.items():
                    (self.root / path).write_text(text)
                self.assertEqual(module.prepare(self.root, bump), expected)
                for path in [
                    "Cargo.toml",
                    "python/Cargo.toml",
                    "pyproject.toml",
                    "python/src/lib.rs",
                ]:
                    self.assertIn(f'"{expected}"', (self.root / path).read_text())
                notes = (self.root / "changelog.md").read_text()
                self.assertIn(f"## {expected} - ", notes)
                self.assertTrue(notes.endswith("## 0.11.0\n\n- Previous\n"))

    def test_invalid_versions_do_not_write(self):
        before = self.snapshot()
        for value in ["0.11.0", "0.10.9", "01.12.0", "0.12.0rc1", "$(id)", "1.2"]:
            with self.subTest(version=value), self.assertRaises(ValueError):
                module.prepare(self.root, "patch", value)
            self.assertEqual(before, self.snapshot())

    def test_explicit_version(self):
        self.assertEqual(module.prepare(self.root, "patch", "2.0.0"), "2.0.0")

    def test_inconsistent_module_does_not_write(self):
        (self.root / "python/src/lib.rs").write_text('m.add("__version__", "0.10.0")?;')
        before = self.snapshot()
        with self.assertRaises(ValueError):
            module.prepare(self.root, "patch")
        self.assertEqual(before, self.snapshot())

    def test_duplicate_release_does_not_write(self):
        with (self.root / "changelog.md").open("a") as stream:
            stream.write("\n## 0.11.1\n")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            module.prepare(self.root, "patch")
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
