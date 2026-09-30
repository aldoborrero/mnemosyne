# mnemosyne

default:
    @just --list

# Run the test suite
test *args:
    pytest tests {{ args }}

# Check Python lint without applying fixes (also enforced by the CI gate)
lint:
    ruff check .

# Apply safe Python lint fixes and format everything
fmt:
    nix fmt

# Every check CI runs — builds the plugin, runs tests, checks lint and formatting
check:
    nix flake check --log-format bar-with-logs

# Build the plugin tree into ./result
build:
    nix build .#mnemosyne --log-format bar-with-logs
