{
  lib,
  buildPythonPackage,
  stdenv,
  fetchFromGitHub,
  darwin,
  cmake,
  huggingface-hub,
  py-cpuinfo,
  scikit-build,
  pytestCheckHook,
}: let
  name = "ctransformers";
  version = "0.2.24";
  isM1 = stdenv.isAarch64 && stdenv.isDarwin;
  osSpecific =
    if isM1
    then with darwin.apple_sdk_11_0.frameworks; [Accelerate]
    else if stdenv.isDarwin
    then with darwin.apple_sdk.frameworks; [Accelerate CoreGraphics CoreVideo]
    else [];
in
  buildPythonPackage {
    inherit version;
    pname = name;
    format = "setuptools";

    src = fetchFromGitHub {
      owner = "marella";
      repo = name;
      rev = "refs/tags/v${version}";
      hash = "sha256-Ub+1z7A4kabQiuL+E2UlHzwY6dFZHSYR4VuFk9ancTY=";
      fetchSubmodules = true;
    };

    propagatedBuildInputs = [
      huggingface-hub
      py-cpuinfo
    ];
    dontUseCmakeConfigure = true;

    nativeBuildInputs = [
      cmake
      scikit-build
    ];
    buildInputs = osSpecific;
    nativeCheckInputs = [pytestCheckHook];
    pythonImportsCheck = ["ctransformers"];
    disabledTestPaths = ["tests/test_model.py"];
    pytestFlagsArray = ["--lib basic"];
    meta = with lib; {
      description = "Python bindings for the Transformer models implemented in C/C++ using GGML library";
      homepage = "https://github.com/marella/ctransformers";
      license = licenses.mit;
    };
  }
