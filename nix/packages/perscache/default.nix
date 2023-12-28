{
  lib,
  buildPythonPackage,
  fetchPypi,
  pytestCheckHook,
  icontract,
  cloudpickle,
  beartype,
  pbr,
  pyyaml,
  pyarrow,
  pandas,
  gcsfs,
}:
buildPythonPackage rec {
  pname = "perscache";
  version = "0.6.1";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-G7L19nuMwTeyxaLMvnx6Pm3F6acrzzVWiaKp8qRRMEc=";
  };
  nativeBuildInputs = [pbr];
  propagatedBuildInputs = [
    icontract
    cloudpickle
    beartype
  ];

  nativeCheckInputs = [
    pytestCheckHook
    pyyaml
    pandas
    pyarrow
    gcsfs
  ];

  meta = with lib; {
    homepage = "https://github.com/leshchenko1979/perscache";
    description = "persistently cache results of functions (or callables in general) using decorators.";
    license = licenses.mit;
  };
}
