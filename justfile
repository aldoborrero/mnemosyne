# mnemosyne

default:
    @just --list

# Run the test suite
test *args:
    pytest tests {{ args }}

# Lint — advisory, not a CI gate (see the ruff note in nix/formatter.nix)
lint:
    ruff check .

# Format everything (nix, python, yaml, toml, markdown, shell)
fmt:
    nix fmt

# Every check CI runs — builds the plugin, runs the tests, checks formatting
check:
    nix flake check --log-format bar-with-logs

# Build the plugin tree into ./result
build:
    nix build .#mnemosyne --log-format bar-with-logs
