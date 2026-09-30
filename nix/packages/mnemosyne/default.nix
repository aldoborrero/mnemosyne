{ flake, pkgs, ... }:
pkgs.callPackage ./package.nix { src = flake; }
