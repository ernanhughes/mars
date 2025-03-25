from dspy import Signature, InputField, OutputField
from dspy import Module, Predict

class FinancialTrendAnalysis(Signature):
    statements = InputField(desc="Multiple income statements across quarters")
    question = InputField(desc="Question about company performance")
    signal = OutputField(desc="bullish, bearish, or neutral")
    rationale = OutputField(desc="Explanation based on financial trends")


class IncomeStatementAnalyzer(Module):
    def __init__(self):
        super().__init__()
        self.analyze = Predict(FinancialTrendAnalysis)

    def forward(self, statements, question):
        return self.analyze(statements=statements, question=question)

if __name__ == "__main__":
    analyzer = IncomeStatementAnalyzer()

    statements = """
    Q1: Revenue = 23B, Operating Income = 2.3B
    Q2: Revenue = 24B, Operating Income = 2.5B
    Q3: Revenue = 25B, Operating Income = 2.4B
    """

    result = analyzer(statements=statements, question="Is the trend bullish?")
    print("📈 Signal:", result.signal)
    print("🧠 Rationale:", result.rationale)
