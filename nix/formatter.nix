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

      # python — formatter only. `ruff check --fix` is deliberately NOT wired in
      # here: ruff 0.16's default rule set flags 288 issues in this tree, 174 of
      # which it would silently rewrite as a side effect of `nix fmt`, and the
      # remaining 114 have no fix — so `nix flake check` could never go green.
      # Linting stays a separate, advisory step (`just lint`), as README says.
      ruff-format.enable = true;

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
