{inputs, ...}: {
  perSystem = {
    config,
    pkgs,
    pkgsCuda,
    pkgsRocm,
    ...
  }: {
    devShells = {
      default =
        (pkgs.python3.withPackages
          (ps: [ps.pytest config.packages.default]))
        .env;
      cuda = pkgsCuda.mkShell {
        packages = [
          pkgsCuda.nixgl.auto.nixGLDefault
          (pkgsCuda.python3.withPackages (ps: [config.packages.cuda ps.pytest]))
        ];
        shellHook = ''
          export LD_LIBRARY_PATH=$(nixGL printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH
        '';
      };
      rocm = pkgsRocm.mkShell {
        packages = [
          pkgsRocm.nixgl.auto.nixGLDefault
          (pkgsRocm.python3.withPackages (ps: [config.packages.rocm ps.pytest]))
        ];
        shellHook = ''
          export LD_LIBRARY_PATH=$(nixGL printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH
        '';
      };
    };
  };
}
