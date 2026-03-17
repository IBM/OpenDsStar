from pathlib import Path

from dotenv import load_dotenv

from agents import OpenDsStarAgent
from agents.utils.model_builder import ModelBuilder
from core.model_registry import ModelRegistry
from ingestion.analyzer import AnalyzerDocumentProcessor
from tools import AnalyzerSummaryRetrievalTool

# Load environment variables
load_dotenv()


def main():

    # Analyzer summary retriever tool example
    query = "what is the life expectancy of men in Afghanistan"

    # Use absolute path relative to the project root
    project_root = Path(__file__).parent.parent.parent
    test_files_dir = project_root / "test" / "files"

    # Code agent analysis
    vector_db, analysis_results = AnalyzerDocumentProcessor(
        llm=ModelRegistry.WX_MISTRAL_MEDIUM,
        embedding_model=ModelRegistry.GRANITE_EMBEDDING,
        db_uri="./milvus_analyzer.db",
    ).process_directory(test_files_dir)

    # Docling + LLM summary analysis
    # vector_db, analysis_results = DoclingDocumentAnalyzer(
    #     llm=ModelRegistry.WX_MISTRAL_MEDIUM,
    #     embedding_model=ModelRegistry.GRANITE_EMBEDDING,
    #     db_uri="./milvus_analyzer.db",
    # ).process_directory(test_files_dir)

    retriever = AnalyzerSummaryRetrievalTool(
        vector_db=vector_db,
        analysis_results=analysis_results,
    )

    # Build model instance before creating agent
    model_instance, _ = ModelBuilder.build(
        model=ModelRegistry.WX_MISTRAL_MEDIUM,
        temperature=0.0,
        cache_dir=Path.home() / ".cache" / "open_ds_star",
        framework="langchain",
    )

    agent = OpenDsStarAgent(model=model_instance, tools=[retriever])

    result = agent.invoke(query)

    # Print results
    print(f"\nQuery: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Steps used: {result['steps_used']}/{result['max_steps']}")
    print(f"LLM calls: {result['num_llm_calls']}")
    print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")


if __name__ == "__main__":
    main()
