{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs =
    { nixpkgs, nixgl, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ nixgl.overlay ];
            config = { allowUnfree = true; };
          };
        in
        {
          default = pkgs.mkShell {
            env = {
              LD_LIBRARY_PATH = with pkgs;lib.makeLibraryPath (pkgs.pythonManylinuxPackages.manylinux1 ++ [
                zstd
                stdenv.cc.cc.lib
                fontconfig
                freetype
                libx11
                glib # libglib-2.0.so.0
                libGL # libGL.so.1
                libxkbcommon # libxkbcommon.so.0
                dbus
                krb5
                libdrm
                xorg.libxcb
                xorg.xcbutilwm
                xorg.xcbutilimage
                xorg.xcbutilkeysyms
                xorg.xcbutilrenderutil
                xcb-util-cursor
                libxcb-util
                wayland
              ]);
              DISPLAY = ":0";
              QT_QPA_PLATFORM = "offscreen";
            };
            buildInputs = with pkgs; [
              mesa
            ];
            packages = with pkgs;[
              python3
              uv
              pkgs.nixgl.auto.nixGLNvidia
              ffmpeg
            ];
            shellHook = ''
              unset PYTHONPATH
              unset QT_PLUGIN_PATH
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}

