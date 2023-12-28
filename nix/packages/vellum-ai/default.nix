{
  buildPythonPackage,
  fetchPypi,
  pythonRelaxDepsHook,
  poetry-core,
  httpx,
  pydantic,
}:
buildPythonPackage rec {
  pname = "vellum-ai";
  version = "0.0.30";
  format = "pyproject";

  src = fetchPypi {
    pname = "vellum_ai";
    inherit version;
    hash = "sha256-iWJtl4WDeXqrCFjDHBBEgkG/PcqLfiJ9QkA92QPlQrY=";
  };

  nativeBuildInputs = [
    poetry-core
    pythonRelaxDepsHook
  ];

  pythonRelaxDeps = [
    "httpx"
  ];

  propagatedBuildInputs = [
    httpx
    pydantic
  ];

  pythonImportsCheck = ["vellum"];

  meta = {
    description = "Product development platform for AI";
    homepage = "https://www.vellum.ai/";
  };
}
