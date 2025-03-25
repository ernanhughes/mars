# Full MARS DSPy Pipeline - Improved Version

import os
from datetime import datetime
import logging
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker

import dspy
from dspy import Signature, InputField, OutputField, Module, Predict, ChainOfThought
from dspy.teleprompt import Teleprompter
from dspy.teleprompt import BootstrapFewShot
from edgar import Company, set_identity
from edgar.xbrl2 import XBRL

# ==== SETUP ====
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ==== DATABASE ====
PG_CONN_STR = os.getenv("PG_CONN_STR")
Base = declarative_base()
engine = create_engine(PG_CONN_STR)
SessionLocal = sessionmaker(bind=engine)

class MarsStep(Base):
    __tablename__ = "mars_steps"
    id = Column(Integer, primary_key=True, index=True)
    input_question = Column(Text)
    teacher_question = Column(Text)
    critique = Column(Text)
    final_question = Column(Text)
    final_answer = Column(Text)
    teacher_latency = Column(Float)
    critic_latency = Column(Float)
    student_latency = Column(Float)
    teacher_agent = Column(String)
    critic_agent = Column(String)
    student_agent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    trace = Column(Text)

def init_db():
    Base.metadata.create_all(bind=engine)

# ==== LOGGING ====
def log_mars_step(input_q, teacher_q, critique, final_q, final_a,
                  t_latency=None, c_latency=None, s_latency=None,
                  t_agent="teacher", c_agent="critic", s_agent="student",
                  trace_json=None):
    session = SessionLocal()
    step = MarsStep(
        input_question=input_q,
        teacher_question=teacher_q,
        critique=critique,
        final_question=final_q,
        final_answer=final_a,
        teacher_latency=t_latency,
        critic_latency=c_latency,
        student_latency=s_latency,
        teacher_agent=t_agent,
        critic_agent=c_agent,
        student_agent=s_agent,
        trace=trace_json
    )
    session.add(step)
    session.commit()
    session.close()

# ==== DSPy CONFIG ====
dspy.configure(lm=dspy.LM('ollama_chat/hf.co/ernanhughes/Fin-R1-Q8_0-GGUF', api_base='http://localhost:11434', api_key=''))

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

# ==== DSPy PROGRAM ====
class MarsAnalysisProgram(dspy.Program):
    def __init__(self, teacher, critic, student):
        super().__init__()
        self.teacher = teacher
        self.critic = critic
        self.student = student

    def forward(self, context: str, base_question: str):
        teacher_out = self.teacher(prompt=context + "\n\n" + base_question)
        critic_out = self.critic(question=teacher_out.question)

        if "yes" in critic_out.critique.lower():
            final_question = f"{base_question} Consider also: {teacher_out.question}"
        else:
            final_question = base_question

        student_out = self.student(context=context, question=final_question)

        return {
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
        self.pg_conn_str = os.getenv("PG_CONN_STR")
        self.identity = os.getenv("IDENTITY")
        self.engine = create_engine(self.pg_conn_str)
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

    def rich_report_to_text(self, df: pd.DataFrame) -> str:
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

# ==== RUN PIPELINE ====
if __name__ == "__main__":
    init_db()

    ticker = "TSLA"
    fetcher = EDGARFetcher(ticker=ticker)
    statements = fetcher.fetch_markdown_statements()
    prompt = build_analysis_prompt(ticker, statements)

    teacher = TeacherQuestioner()
    critic = CriticJudge()
    student = MarginAnalyzer()

    program = MarsAnalysisProgram(teacher, critic, student)
    results = program(
        context=prompt,
        base_question="Is the company improving its profitability?"
    )
    print(results)


