from edgar_loader import load_sample_income_statement
from haystack_pipeline import build_haystack_pipeline
from smol_agent import build_trading_agent

from haystack.schema import Document

def main():
    # Load sample income statement
    statement = load_sample_income_statement()
    doc = Document(content=statement["data"], meta={"company": statement["company"], "year": statement["year"]})

    # Build pipeline and agent
    pipeline = build_haystack_pipeline([doc])
    agent = build_trading_agent(pipeline)

    # Run agent
    question = "Is ACME Corp improving profitability?"
    response = agent.run(question)
    print("\\n🧠 Agent Answer:")
    print(response)

if __name__ == "__main__":
    main()
