{ pkgs, perSystem }:
perSystem.devshell.mkShell {
  packages = (
    with pkgs;
    [
      just
      perSystem.self.formatter

      # pyproject.toml floors at 3.11. `python3` is whatever the pinned nixpkgs
      # carries, which is exactly what package.nix builds and tests against —
      # naming the same attribute in both keeps the shell and the build in step.
      python3

      # The test suite stubs Hermes and both memory backends (tests/conftest.py),
      # so plain pytest is all it needs — no venv, no network.
      python3Packages.pytest

      # Same binary treefmt drives, for the fast `just lint` loop.
      ruff

      # Only needed to pull the two runtime deps (honcho-ai, hindsight-client)
      # into $PRJ_ROOT/.venv — neither is packaged in nixpkgs, so they come from
      # PyPI when you want to exercise the plugin against a real Hermes.
      uv
    ]
  );

  env = [
    {
      name = "NIX_PATH";
      value = "nixpkgs=${toString pkgs.path}";
    }
    {
      name = "NIX_DIR";
      eval = "$PRJ_ROOT/nix";
    }
    {
      # Keep the toolchain cache in the project rather than $HOME.
      name = "UV_PROJECT_ENVIRONMENT";
      eval = "$PRJ_ROOT/.venv";
    }
  ];
}
