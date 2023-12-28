{
  lib,
  buildPythonPackage,
  pythonOlder,
  fetchPypi,
  setuptools-scm,
  jinja2,
  interegular,
  lark,
  nest-asyncio,
  numpy,
  perscache,
  pydantic,
  scipy,
  torch,
  numba,
  joblib,
  referencing,
  jsonschema,
  requests,
}:
buildPythonPackage rec {
  pname = "outlines";

  version = "0.0.18";
  disabled = pythonOlder "3.8";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-RMRnCLxT83YUzBY5LiExsXZ32Te5iaX30EYjc/ID4Yc=";
  };
  buildInputs = [setuptools-scm];

  propagatedBuildInputs = [
    jinja2
    interegular
    lark
    nest-asyncio
    numpy
    perscache
    pydantic
    scipy
    torch
    numba
    joblib
    referencing
    jsonschema
    requests
  ];
  doCheck = false;
  meta = with lib; {
    description = "Robust (guided) text generation";
    homepage = "https://github.com/outlines-dev/outlines";
    license = licenses.mit;
  };
}
