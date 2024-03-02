{
  lib,
  buildPythonPackage,
  pythonOlder,
  setuptools,
  dynaconf,
  numpy,
  watchdog,
  xdg,
  accelerate,
  transformers,
  ctransformers,
  flask,
  flask-cors,
  llama-index,
  levenshtein,
  beautifulsoup4,
  sentencepiece,
  sentence-transformers,
  requests,
  spacy,
  outlines,
  gradio,
  backoff,
  peft,
  pinecone-client,
  pypdf2,
  trl,
  weaviate-client,
  python-dotenv,
}:
buildPythonPackage {
  pname = "llm_vm";
  version = "0.1.55";
  format = "pyproject";
  disabled = pythonOlder "3.10";
  src = ../.;
  buildInputs = [
    setuptools
  ];
  propagatedBuildInputs = [
    dynaconf
    numpy
    watchdog
    xdg
    accelerate
    transformers
    ctransformers
    flask
    flask-cors
    llama-index
    levenshtein
    beautifulsoup4
    sentencepiece
    sentence-transformers
    requests
    spacy
    outlines
    gradio
    backoff
    peft
    pinecone-client
    pypdf2
    trl
    weaviate-client
    python-dotenv
  ];

  # Tests use HuggingFace which attempts to create directories and access the internet
  doCheck = false;

  meta = with lib; {
    description = "Strongly typed, zero-effort CLI interfaces";
    homepage = "https://github.com/brentyi/tyro";
    license = licenses.mit;
    maintainers = let
      collinarnett = {
        name = "Collin Arnett";
        email = "collin@arnett.it";
        github = "collinarnett";
        githubId = 38230482;
      };
    in [collinarnett];
  };
}
