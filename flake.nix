{
  description = "Anarchy AI - A highly optimized and opinionated backend for running LLMs";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixgl.url = "github:guibou/nixGL";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };
  nixConfig.extra-substituters = [
    "https://cuda-maintainers.cachix.org"
    "https://nixos-rocm.cachix.org"
  ];
  nixConfig.extra-trusted-public-keys = [
    "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    "nixos-rocm.cachix.org-1:VEpsf7pRIijjd8csKjFNBGzkBqOmw8H9PRmgAq14LnE="
  ];
  outputs = {
    self,
    nixpkgs,
    flake-parts,
    nixgl,
  } @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;}
    {
      systems = ["x86_64-linux"];
      imports = [
        ./nix/nixpkgs-instances.nix
        ./nix/apps.nix
        ./nix/devshells.nix
      ];
      perSystem = {
        config,
        pkgs,
        pkgsCuda,
        pkgsRocm,
        ...
      }: {
        packages = {
          default = (pkgs.callPackage ./nix {}).llm-vm;
          cuda = (pkgsCuda.callPackage ./nix {}).llm-vm;
          rocm = (pkgsRocm.callPackage ./nix {}).llm-vm;
        };
      };
    };
}
