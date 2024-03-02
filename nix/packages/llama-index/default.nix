{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  pytestCheckHook,
  pythonRelaxDepsHook,
  poetry-core,
  nltk,
  pillow,
  beautifulsoup4,
  faiss,
  fsspec,
  langchain,
  nest-asyncio,
  numpy,
  openai,
  pandas,
  sqlalchemy,
  tiktoken,
  vellum-ai,
}:
buildPythonPackage rec {
  pname = "llama-index";
  version = "0.9.17";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "run-llama";
    repo = "llama_index";
    rev = "refs/tags/v${version}";
    hash = "sha256-bmqucvIYStRak0qBWG0XihSm+zJqDo8GrhbViIK+ffU=";
  };

  nativeBuildInputs = [
    pythonRelaxDepsHook
    poetry-core
  ];
  nativeCheckInputs = [
    pytestCheckHook
    nltk
    pillow
  ];

  pythonRelaxDeps = [
    "fsspec"
    "langchain"
    "sqlalchemy"
  ];

  propagatedBuildInputs = [
    beautifulsoup4
    faiss
    fsspec
    langchain
    nest-asyncio
    numpy
    openai
    pandas
    sqlalchemy
    tiktoken
    vellum-ai
  ];

  disabledTestPaths = [
    "tests/agent/openai/test_openai_agent.py"
    "tests/agent/react/test_react_agent.py"
    "tests/callbacks/test_token_counter.py"
    "tests/chat_engine/test_condense_question.py"
    "tests/chat_engine/test_simple.py"
    "tests/embeddings/test_base.py"
    "tests/embeddings/test_utils.py"
    "tests/indices/document_summary/test_index.py"
    "tests/indices/document_summary/test_retrievers.py"
    "tests/indices/empty/test_base.py"
    "tests/indices/keyword_table/test_base.py"
    "tests/indices/keyword_table/test_retrievers.py"
    "tests/indices/keyword_table/test_utils.py"
    "tests/indices/knowledge_graph/test_base.py"
    "tests/indices/knowledge_graph/test_retrievers.py"
    "tests/indices/list/test_index.py"
    "tests/indices/list/test_retrievers.py"
    "tests/indices/postprocessor/test_base.py"
    "tests/indices/postprocessor/test_llm_rerank.py"
    "tests/indices/postprocessor/test_optimizer.py"
    "tests/indices/query/query_transform/test_base.py"
    "tests/indices/query/test_compose_vector.py"
    "tests/indices/query/test_compose.py"
    "tests/indices/query/test_query_bundle.py"
    "tests/indices/response/test_response_builder.py"
    "tests/indices/response/test_tree_summarize.py"
    "tests/indices/struct_store/test_base.py"
    "tests/indices/struct_store/test_json_query.py"
    "tests/indices/struct_store/test_sql_query.py"
    "tests/indices/test_loading_graph.py"
    "tests/indices/test_loading.py"
    "tests/indices/test_node_utils.py"
    "tests/indices/test_prompt_helper.py"
    "tests/indices/test_utils.py"
    "tests/indices/tree/test_embedding_retriever.py"
    "tests/indices/tree/test_index.py"
    "tests/indices/tree/test_retrievers.py"
    "tests/indices/vector_store/test_deeplake.py"
    "tests/indices/vector_store/test_faiss.py"
    "tests/indices/vector_store/test_pinecone.py"
    "tests/indices/vector_store/test_retrievers.py"
    "tests/indices/vector_store/test_simple.py"
    "tests/llm_predictor/vellum/test_predictor.py"
    "tests/llm_predictor/vellum/test_prompt_registry.py"
    "tests/llms/test_openai.py"
    "tests/llms/test_palm.py"
    "tests/memory/test_chat_memory_buffer.py"
    "tests/objects/test_base.py"
    "tests/playground/test_base.py"
    "tests/query_engine/test_pandas.py"
    "tests/question_gen/test_llm_generators.py"
    "tests/selectors/test_llm_selectors.py"
    "tests/test_utils.py"
    "tests/text_splitter/test_code_splitter.py"
    "tests/text_splitter/test_sentence_splitter.py"
    "tests/token_predictor/test_base.py"
    "tests/tools/test_ondemand_loader.py"
  ];

  # pythonImportsCheck = [ "llama_index" ];
  doCheck = false;

  meta = with lib; {
    description = "Interface between LLMs and your data";
    homepage = "https://github.com/jerryjliu/llama_index";
    license = licenses.mit;
  };
}
