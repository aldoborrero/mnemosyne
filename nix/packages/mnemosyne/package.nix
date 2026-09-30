{
  lib,
  src,
  stdenvNoCC,
  python3,
}:
# Mnemosyne is consumed as a plugin *directory* under $HERMES_HOME/plugins, not
# as an installed distribution: its modules use package-relative imports that
# only resolve under the Hermes plugin loader, and pyproject.toml's flat
# `py-modules` layout would install `cli`, `config`, … as top-level modules that
# cannot import each other. So this installs the tree and runs the test suite,
# rather than building a wheel.
stdenvNoCC.mkDerivation {
  pname = "mnemosyne";
  version = "0.1.0";

  inherit src;

  dontConfigure = true;
  dontBuild = true;

  # The plugin tree is delivered verbatim: install.sh is meant to run on the
  # user's machine against their Hermes venv, so its `/usr/bin/env bash` shebang
  # must survive rather than be rewritten to a store path.
  dontPatchShebangs = true;

  # honcho-ai and hindsight-client are *runtime* deps reached through Hermes
  # (`plugins.memory.hindsight`), never imported directly by this tree — and
  # neither is in nixpkgs. install.sh pulls them into the Hermes venv; the test
  # suite stubs them, so the build needs nothing but pytest.
  doCheck = true;
  nativeCheckInputs = [ (python3.withPackages (ps: [ ps.pytest ])) ];
  checkPhase = ''
    runHook preCheck
    pytest tests
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    install -d "$out/share/hermes/plugins/mnemosyne"
    cp -r -- *.py plugin.yaml install.sh README.md README.ru.md LICENSE \
      "$out/share/hermes/plugins/mnemosyne/"
    runHook postInstall
  '';

  meta = {
    description = "Composite long-term memory plugin for Hermes Agent — Honcho (user model) + Hindsight (facts)";
    homepage = "https://github.com/aldoborrero/mnemosyne";
    license = lib.licenses.asl20;
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
    maintainers = [ ];
    platforms = lib.platforms.all;
  };
}
