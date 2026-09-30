{
  flake,
  inputs,
  pkgs,
  ...
}:
let
  mod = inputs.treefmt-nix.lib.evalModule pkgs {
    projectRootFile = "flake.nix";

    programs = {
      # nix — deadnix strips dead code, then nixfmt formats
      deadnix.enable = true;
      deadnix.no-lambda-pattern-names = true; # don't break callPackage-style { a, b, ... } patterns
      deadnix.priority = 1;
      nixfmt.enable = true;
      nixfmt.priority = 2;

      # python — safe lint fixes first, then formatting; rules in pyproject.toml
      ruff-check.enable = true;
      ruff-check.priority = 1;
      ruff-format.enable = true;
      ruff-format.priority = 2;

      # shell — install.sh and .envrc are not *.sh, so they need `includes`
      shellcheck = {
        enable = true;
        includes = [
          "*.sh"
          "*.bash"
          "*.envrc"
          "*.envrc.*"
        ];
        priority = 1;
      };
      shfmt = {
        enable = true;
        includes = [
          "*.sh"
          "*.bash"
          "*.envrc"
          "*.envrc.*"
        ];
        priority = 2;
      };

      yamlfmt.enable = true;
      yamlfmt.settings.formatter = {
        type = "basic";
        indent = 2;
        retain_line_breaks = true;
      };
      jsonfmt.enable = true;
      taplo.enable = true; # TOML
      mdformat.enable = true; # Markdown
      just.enable = true;
    };
  };
  wrapper = mod.config.build.wrapper;
in
wrapper
// {
  passthru = wrapper.passthru // {
    tests.check = mod.config.build.check flake;
  };
}
