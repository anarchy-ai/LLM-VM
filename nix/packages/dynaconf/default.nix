{
  lib,
  buildPythonPackage,
  fetchPypi,
  pythonOlder,
  pytestCheckHook,
  setuptools,
  flask,
  django,
  configobj,
  redis,
  hvac,
  ruamel-yaml,
}:
buildPythonPackage rec {
  pname = "dynaconf";
  version = "3.2.4";
  format = "setuptools";

  disabled = pythonOlder "3.8";
  doCheck = false;

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-LmreuqWH9N+SQaFqS+w/2lIRVNJrFfMlj951OlkoMbY=";
  };

  propagatedBuildInputs = [setuptools];

  nativeCheckInputs = [pytestCheckHook configobj flask django];
  passthru.optional-dependencies = {
    ini = [configobj];
    configobj = [configobj];
    yaml = [ruamel-yaml];
    vault = [hvac];
    redis = [redis];
    all = [redis configobj hvac ruamel-yaml];
  };

  disabledTestPaths = ["tests/test_redis.py" "tests/test_vault.py"];

  pythonImportsCheck = [
    "dynaconf"
  ];

  meta = with lib; {
    description = "The dynamic configurator for your Python Project";
    homepage = "https://github.com/dynaconf/dynaconf";
    license = licenses.mit;
  };
}
