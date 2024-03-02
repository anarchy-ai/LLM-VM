{
  lib,
  pkgs,
  ...
}: let
  callPackage = lib.callPackageWith (pkgs // pkgs.python3Packages // packages);
  packages = {
    ctransformers = callPackage ./packages/ctransformers {};
    dynaconf = callPackage ./packages/dynaconf {};
    interegular = callPackage ./packages/interegular {};
    vellum-ai = callPackage ./packages/vellum-ai {};
    llama-index = callPackage ./packages/llama-index {};
    outlines = callPackage ./packages/outlines {};
    perscache = callPackage ./packages/perscache {};
    trl = callPackage ./packages/trl {};
    tyro = callPackage ./packages/tyro {};
    llm-vm = callPackage ./package.nix {};
  };
in
  packages
