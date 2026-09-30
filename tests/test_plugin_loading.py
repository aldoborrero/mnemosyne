"""Exercise Hermes' file-based discovery without installing Hermes or backends."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("first", ["provider", "cli"])
@pytest.mark.parametrize("placeholder", [False, True])
@pytest.mark.parametrize(
    "package_name",
    ["_hermes_user_memory.mnemosyne", "_hermes_user_memory.mnemosyne__source_test"],
)
def test_directory_discovery(first, placeholder, package_name, tmp_path):
    plugin_dir = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            r"""
import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path

root = Path(sys.argv[1])
name, first, placeholder = sys.argv[2:]
agent = types.ModuleType("agent")
base = types.ModuleType("agent.memory_provider")
base.MemoryProvider = type("MemoryProvider", (), {})
sys.modules["agent"] = agent
sys.modules["agent.memory_provider"] = base

if placeholder == "True":
    parent = types.ModuleType("_hermes_user_memory")
    parent.__path__ = []
    sys.modules[parent.__name__] = parent
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package

def load(module_name, file):
    spec = importlib.util.spec_from_file_location(module_name, root / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

if first == "cli":
    cli = load(name + ".cli", "cli.py")
    package = sys.modules[name]
else:
    package = load(name, "__init__.py")
    cli = load(name + ".cli", "cli.py")

provider = importlib.import_module(name + ".provider")
commands = importlib.import_module(name + ".commands")
config = importlib.import_module(name + ".config")
assert package.MnemosyneMemoryProvider is provider.MnemosyneMemoryProvider
assert package.register is provider.register
assert commands.config is provider.config is config
assert Path(config.__file__).parent == root / "src" / "mnemosyne"
assert sys.modules[name] is package
assert not any(key.startswith("mnemosyne.") for key in sys.modules)

parser = argparse.ArgumentParser()
cli.register_cli(parser)
args = parser.parse_args(["status"])
assert args.func is cli.mnemosyne_command is commands.mnemosyne_command

policy = importlib.import_module(name + ".policy")
approved = importlib.import_module(name + ".approved")
seen = []
context = types.SimpleNamespace(register_memory_provider=seen.append)
composite_instance, approved_instance = object(), object()
provider.MnemosyneMemoryProvider = lambda: composite_instance
approved.ApprovedMemoryProvider = lambda: approved_instance
policy.approved_writes_only = lambda: False
package.register(context)
policy.approved_writes_only = lambda: True
package.register(context)
assert seen == [composite_instance, approved_instance]
""",
            str(plugin_dir),
            package_name,
            first,
            str(placeholder),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
