{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  pythonOlder,
  docstring-parser,
  typing-extensions,
  backports-cached-property,
  colorama,
  rich,
  shtab,
}:
buildPythonPackage rec {
  pname = "tyro";
  version = "0.6.0";
  format = "pyproject";
  disabled = pythonOlder "3.7";
  src = fetchFromGitHub {
    owner = "brentyi";
    repo = pname;
    rev = "refs/tags/v${version}";
    hash = "sha256-7FZ22CsyNSVN3Nr2BH7GxKLgmNjN9s/34j+vbzvF2aA=";
  };

  propagatedBuildInputs = [
    docstring-parser
    typing-extensions
    backports-cached-property
    colorama
    rich
    shtab
  ];

  doCheck = false;
  meta = with lib; {
    description = "Strongly typed, zero-effort CLI interfaces";
    homepage = "https://github.com/brentyi/tyro";
    license = licenses.mit;
  };
}
