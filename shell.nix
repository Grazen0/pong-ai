{
  pkgs ? import <nixpkgs> { },
  ...
}:
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    catch2_3
  ];

  env = {
    CMAKE_PREFIX_PATH = "${pkgs.catch2_3}/lib/cmake";
  };
}
