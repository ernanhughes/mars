from pathlib import Path

# Recreate directory structure and files after code state reset
base_dir = Path("/mnt/data/edgar_smolagents_haystack")
base_dir.mkdir(parents=True, exist_ok=True)

files = {
    "edgar_loader.py": """
def load_sample_income_statement():
    return {
        "company": "ACME Corp",
        "year": 2023,
        "data": '''
        Income Statement (in millions USD)
        Revenue: 500
        Cost of Goods Sold: 300
        Gross Profit: 200
        Operating Expenses: 100
        Operating Income: 100
        Net Income: 80
        '''
    }
""",
    "haystack_pipeline.py": """
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import DocumentSearchPipeline

def build_haystack_pipeline(docs):
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)

    retriever = BM25Retriever(document_store=document_store)
    pipeline = DocumentSearchPipeline(retriever)

    return pipeline
""",
    "smol_agent.py": """
from smolagent import Agent

def build_trading_agent(pipeline):
    def query_tool(query: str) -> str:
        result = pipeline.run(query=query)
        docs = result['documents']
        return '\\n'.join([d.content for d in docs])

    agent = Agent(
        tools={"financial_search": query_tool},
        prompt=\"\"\"
You are a financial analyst agent. Use the 'financial_search' tool to retrieve financial data
and answer the user's trading question based on the company's income statement.
\"\"\"
    )
    return agent
""",
    "main.py": """
"""
}

# Write all files to disk
for filename, content in files.items():
    (base_dir / filename).write_text(content.strip())

base_dir.name
