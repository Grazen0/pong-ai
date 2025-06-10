{
  pkgs ? import <nixpkgs> { },
  ...
}:
let
  inherit (pkgs)
    lib
    cmake
    sdl3
    sdl3-ttf
    catch2_3
    xxd
    ;
in
pkgs.stdenv.mkDerivation {
  pname = "pong-ai";
  version = "0.1.0";

  src = lib.cleanSource ./.;

  nativeBuildInputs = [
    cmake
    sdl3.dev
    sdl3-ttf
    catch2_3
    xxd
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DTESTING_BUILD=on"
  ];

  enableParallelBuilding = true;
  doCheck = true;

  installPhase = ''
    runHook preInstall
    install -Dm755 pong_ai -t "$out/bin"
    runHook postInstall
  '';

  meta = with lib; {
    description = "A pong AI agent written in C++";
    homepage = "https://github.com/Grazen0/pong-ai";
    license = licenses.gpl3;
    mainProgram = "pong_ai";
  };
}
