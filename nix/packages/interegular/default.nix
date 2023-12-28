{
  lib,
  buildPythonPackage,
  unittestCheckHook,
  fetchPypi,
  setuptools,
  wheel,
}:
buildPythonPackage rec {
  pname = "interegular";
  version = "0.3.2";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-6kZFq6zpPZ8500EgVoamC3l3ekPZRACTPv+5yPKQGCU=";
  };
  nativeBuildInputs = [
    setuptools
    wheel
  ];
  naitiveCheckInputs = [
    unittestCheckHook
  ];
  #postCheck = ''
  #  PYTHONPATH=$out/${python.sitePackages}:PYTHONPATH
  #  python -c "from interegular.fsm import Alphabet"
  #'';
  meta = with lib; {
    homepage = "https://github.com/MegaIng/interegular";
    description = "regex intersection checker";
    license = licenses.mit;
  };
}
