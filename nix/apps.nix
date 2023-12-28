{
  perSystem = {
    config,
    lib,
    pkgs,
    pkgsCuda,
    pkgsRocm,
    ...
  }: {
    apps = let
      wrapNixGL = pkgs: backend: app: name: {
        type = "app";
        program = let
          p = pkgs.writeShellScriptBin "${name}-${backend}" ''

            #!/bin/sh
            ${pkgs.nixgl.auto.nixGLDefault}/bin/nixGL ${app}/bin/${name} "$@"
          '';
        in "${p}/bin/${name}-${backend}";
      };

      mkDefault = name: {
        type = "app";
        program = "${config.packages.default}/bin/${name}";
      };

      wrappedPackagesFns = {
        cuda = name: backend: wrapNixGL pkgsCuda backend config.packages.cuda name;
        rocm = name: backend: wrapNixGL pkgsRocm backend config.packages.rocm name;
      };

      binaries = [
        "llm_vm_run_agent"
        "llm_vm_run_agent_backwards_chaining"
        "llm_vm_run_agent_flat"
        "llm_vm_server"
        "llm_vm_run_agent_rebel"
      ];
    in
      (lib.foldl' (a: b: a // b) {} (builtins.map (name:
        lib.mapAttrs'
        (backend: f:
          lib.nameValuePair "${name}-${backend}" (f name backend))
        wrappedPackagesFns)
      binaries))
      // (lib.genAttrs binaries mkDefault)
      // {
        default = {
          type = "app";
          program = "${config.packages.default}/bin/llm_vm_server";
        };
      };
  };
}
