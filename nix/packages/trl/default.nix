{
  lib,
  buildPythonPackage,
  pythonOlder,
  fetchFromGitHub,
  datasets,
  torch,
  tqdm,
  transformers,
  accelerate,
  peft,
  tyro,
}:
buildPythonPackage rec {
  pname = "trl";
  version = "0.7.4";
  format = "pyproject";
  disabled = pythonOlder "3.7";
  src = fetchFromGitHub {
    owner = "huggingface";
    repo = pname;
    rev = "refs/tags/v${version}";
    hash = "sha256-wUmKuA893VWvapmwCl+YwdIu/8H5KLVN42NGQrXKljA=";
  };

  propagatedBuildInputs = [
    datasets
    torch
    tqdm
    transformers
    accelerate
    peft
    tyro
  ];

  doCheck = false;
  meta = with lib; {
    description = "Full stack transformer language models with reinforcement learning.";
    homepage = "https://github.com/huggingface/trl";
    license = licenses.asl20;
  };
}
