import os
import pandas as pd
import socket
import dspy
from dspy import Signature, InputField, OutputField, Module, Predict, ChainOfThought, LM
from edgar import Company, set_identity
from edgar.xbrl2 import XBRL

import litellm
litellm._turn_on_debug()
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('mars.log', 'w', 'utf-8')])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ==== DSPy CONFIG ====
# Check if running on Hugging Face Spaces
running_in_spaces = os.getenv("SYSTEM") == "spaces" or "hf.space" in socket.getfqdn()
if running_in_spaces:
    print("🔍 Detected: Running in Hugging Face Spaces")
    dspy.configure(
        lm=LM(
            model='huggingface/SUFE-AIFLM-Lab/Fin-R1',
            api_base='https://api-inference.huggingface.co',
            api_key=os.getenv("HF_API_KEY")
        )
    )
else:
    print("💻 Detected: Running locally")
    dspy.configure(
        lm=LM(
            model='ollama_chat/hf.co/ernanhughes/Fin-R1-Q8_0-GGUF',
            api_base='http://localhost:11434',
            api_key=''  # Ollama does not require key
        )
    )


# ==== DSPy SIGNATURES ====
class AnalyzeMargins(Signature):
    context = InputField()
    question = InputField()
    signal = OutputField()
    rationale = OutputField()

class FinancialTrendAnalysis(Signature):
    statements = InputField()
    question = InputField()
    signal = OutputField()
    rationale = OutputField()

class PlannerSignature(Signature):
    base_question = InputField()
    steps = OutputField(desc="List of reasoning substeps to answer the question")

# ==== DSPy MODULES ====
class IncomeStatementAnalyzer(Module):
    def __init__(self):
        super().__init__()
        self.analyze = Predict(FinancialTrendAnalysis)

    def forward(self, statements, question):
        return self.analyze(statements=statements, question=question)

class TeacherQuestion(Signature):
    prompt = InputField()
    question = OutputField()

class TeacherQuestioner(Module):
    def __init__(self, use_chain_of_thought: bool = True):
        super().__init__()
        self.generate = ChainOfThought(TeacherQuestion) if use_chain_of_thought else Predict(TeacherQuestion)

    def forward(self, prompt):
        return self.generate(prompt=prompt)

class CritiqueQuestion(Signature):
    question = InputField()
    critique = OutputField()

class CriticJudge(Module):
    def __init__(self):
        super().__init__()
        self.evaluate = Predict(CritiqueQuestion)

    def forward(self, question):
        return self.evaluate(question=question)

class MarginAnalyzer(Module):
    def __init__(self):
        super().__init__()
        self.analyze = ChainOfThought(AnalyzeMargins)

    def forward(self, context, question, teacher_question=None):
        if teacher_question:
            question = f"{question} Consider also: {teacher_question}"
        return self.analyze(context=context, question=question)

class PlannerModule(Module):
    def __init__(self):
        super().__init__()
        self.plan = ChainOfThought(PlannerSignature)

    def forward(self, base_question):
        return self.plan(base_question=base_question)

# ==== DSPy PROGRAM ====
class MarsAnalysisProgram(dspy.Program):
    def __init__(self, planner, teacher, critic, student):
        super().__init__()
        self.planner = planner
        self.teacher = teacher
        self.critic = critic
        self.student = student

    def forward(self, context: str, base_question: str):
        plan_out = self.planner(base_question=base_question)
        teacher_out = self.teacher(prompt=context + "\n\n" + base_question)
        critic_out = self.critic(question=teacher_out.question)

        if "yes" in critic_out.critique.lower():
            final_question = f"{base_question} Consider also: {teacher_out.question}"
        else:
            final_question = base_question

        student_out = self.student(context=context, question=final_question)

        return {
            "plan": plan_out.steps,
            "teacher_question": teacher_out.question,
            "critique": critic_out.critique,
            "final_question": final_question,
            "signal": student_out.signal,
            "rationale": student_out.rationale
        }

# ==== UTILS ====
def estimate_token_count(markdown_list: list[str], chars_per_token: int = 4) -> int:
    combined_text = "\n\n".join(markdown_list)
    return len(combined_text) // chars_per_token

def build_analysis_prompt(ticker: str, markdown_list: list[str]) -> str:
    header = f"You are a financial analysis model. Below are the last {len(markdown_list)} income statements from {ticker}.\n\n"
    instructions = (
        "Analyze the trend in revenue and operating income.\n"
        "Decide if profitability is improving or declining.\n"
        "Then provide a trading signal.\n\n"
        "Respond with:\n"
        "Signal: <Bullish/Bearish/Neutral>\n"
        "Rationale: <short explanation>\n\n"
    )
    body = "\n\n".join(markdown_list)
    return header + instructions + body

# ==== EDGAR FETCHER ====
class EDGARFetcher:
    def __init__(self, ticker: str, form: str = "10-Q", n: int = 3):
        self.identity = "marsgradioapp@gmail.com"
        self.ticker = ticker
        self.form = form
        self.n = n
        set_identity(self.identity)

    def fetch_markdown_statements(self):
        filings = Company(self.ticker).latest(form=self.form, n=self.n)
        statements = []
        for filing in filings:
            xbrl = XBRL.from_filing(filing)
            income_statement = xbrl.statements.income_statement()
            df = income_statement.to_dataframe()
            statements.append(self.rich_report_to_text(df))
        return statements

    @staticmethod
    def rich_report_to_text(df: pd.DataFrame) -> str:
        lines = []
        for _, row in df.iterrows():
            label = row.get("original_label") or row.get("label") or row.get("concept")
            values = [
                f"{col}: {row[col]}" for col in df.columns
                if isinstance(col, str) and col.startswith("20") and pd.notna(row[col])
            ]
            if values:
                lines.append(f"{label}: " + " | ".join(values))
        return "\n".join(lines)

def analyze_ticker(ticker: str):
    """
    Run the full MARS analysis pipeline for a given stock ticker.

    Args:
        ticker (str): Stock symbol (e.g. 'TSLA')

    Returns:
        dict: MARS pipeline result containing plan, teacher_question, critique,
              final_question, signal, and rationale
    """
    fetcher = EDGARFetcher(ticker=ticker)
    statements = fetcher.fetch_markdown_statements()
    prompt = build_analysis_prompt(ticker, statements)

    planner = PlannerModule()
    teacher = TeacherQuestioner()
    critic = CriticJudge()
    student = MarginAnalyzer()

    program = MarsAnalysisProgram(planner, teacher, critic, student)
    result = program(
        context=prompt,
        base_question="Is the company improving its profitability?"
    )
    logger.info(f"Result for stock {ticker}:\n{result}")

    return result

