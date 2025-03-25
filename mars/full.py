from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()
import dspy
from dspy import Signature, InputField, OutputField, Module
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

from edgar import Company, set_identity
from edgar.xbrl2 import *


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

def init_db():
    Base.metadata.create_all(bind=engine)



def log_mars_step(input_q, teacher_q, critique, final_q, final_a,
                  t_latency=None, c_latency=None, s_latency=None,
                  t_agent="teacher", c_agent="critic", s_agent="student"):
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
        student_agent=s_agent
    )
    session.add(step)
    session.commit()
    session.close()




lm = dspy.LM('ollama_chat/hf.co/ernanhughes/Fin-R1-Q8_0-GGUF', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# Step 1: Define DSPy Signature
class AnalyzeMargins(Signature):
    context = InputField(desc="Relevant financial data")
    question = InputField(desc="User's trading question")
    signal = OutputField(desc="Bullish, Bearish, or Neutral")
    rationale = OutputField(desc="Explanation for the signal")
    
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



class EDGARFetcher:
    def __init__(self, ticker: str, form: str = "10-Q", n: int = 3):
        load_dotenv()

        # Load PG and EDGAR credentials
        self.pg_conn_str = os.getenv("PG_CONN_STR")
        self.identity = os.getenv("IDENTITY")
        self.engine = create_engine(self.pg_conn_str)
        self.ticker = ticker
        self.form = form
        self.n = n

        # Set identity for SEC API
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
        """
        Convert a rich EDGAR report DataFrame to readable plain text for LLMs.
        """
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


    def save_statements_to_db(self, statement_list, table="filings_markdown"):
        for statement in statement_list:
            df = statement.to_dataframe()
            df = df.T
            df = df.reset_index()
            print(df.head())
            df.to_sql(table, con=self.engine, index=False, if_exists="append")
            print(f"Inserted {len(df)} rows into {table}")

    def run(self):
        statements = self.fetch_markdown_statements()
        # self.save_statements_to_db(statements, table="income_statement")
        markdowns = [statement for statement in statements]
        return markdowns
ticker = "TSLA"
n = 3
fetcher = EDGARFetcher(ticker=ticker, n=n)
statements = fetcher.run()

# Create a utility function to estimate token count from a list of markdown statements

def estimate_token_count(markdown_list: list[str], chars_per_token: int = 4) -> int:
    """
    Estimate the number of tokens used by a list of markdown-formatted statements.

    Args:
        markdown_list (list[str]): A list of markdown text blocks.
        chars_per_token (int): Average number of characters per token. Default is 4.

    Returns:
        int: Estimated total token count.
    """
    combined_text = "\n\n".join(markdown_list)
    total_chars = len(combined_text)
    estimated_tokens = total_chars // chars_per_token
    return estimated_tokens

estimated = estimate_token_count(statements)
print(estimated)


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

prompt = build_analysis_prompt(ticker, statements)
print(prompt[:300])

import litellm
litellm._turn_on_debug()
import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler('mars.log', 'w', 'utf-8')])


from dspy import Signature, InputField, OutputField, Module

from dspy import Module, Predict, ChainOfThought
from dspy import Signature, InputField, OutputField

class TeacherQuestion(Signature):
    prompt = InputField()
    question = OutputField(desc="A Socratic question to improve the prompt")

class TeacherQuestioner(Module):
    def __init__(self, use_chain_of_thought: bool = True):
        super().__init__()
        self.use_chain_of_thought = use_chain_of_thought

        self.generate = (
            ChainOfThought(TeacherQuestion)
            if self.use_chain_of_thought
            else Predict(TeacherQuestion)
        )

    def forward(self, prompt):
        return self.generate(prompt=prompt)


class CritiqueQuestion(Signature):
    question = InputField()
    critique = OutputField(desc="Is the question Socratic? Why or why not?")

class CriticJudge(Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Predict(CritiqueQuestion)

    def forward(self, question):
        return self.evaluate(question=question)

class MarginAnalyzer(Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzeMargins)

    def forward(self, context, question, teacher_question=None):
        if teacher_question:
            question = f"{question} Consider also: {teacher_question}"
        return self.analyze(context=context, question=question)

import dspy

class MarsAnalysisProgram(dspy.Program):
    def __init__(self, teacher: dspy.Module, critic: dspy.Module, student: dspy.Module):
        super().__init__()
        self.teacher = teacher
        self.critic = critic
        self.student = student

    def forward(self, context: str, base_question: str):
        # Step 1: Generate Socratic question
        teacher_out = self.teacher(prompt=context + "\n\n" + base_question)
        
        # Step 2: Evaluate Socratic quality
        critic_out = self.critic(question=teacher_out.question)

        # Step 3: Decide final question
        if "yes" in critic_out.critique.lower():
            final_question = f"{base_question} Consider also: {teacher_out.question}"
        else:
            final_question = base_question

        # Step 4: Get Student’s answer
        student_out = self.student(context=context, question=final_question)

        return {
            "teacher_question": teacher_out.question,
            "critique": critic_out.critique,
            "final_question": final_question,
            "signal": student_out.signal,
            "rationale": student_out.rationale
        }

def log_mars_step(input_q, teacher_q, critique, final_q, final_a,
                  t_latency=None, c_latency=None, s_latency=None,
                  t_agent="teacher", c_agent="critic", s_agent="student",
                  trace_json=None):
    # from db import SessionLocal, MarsStep

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
        trace=trace_json  # ➕ this assumes you added a 'trace' TEXT/JSON column to your table
    )
    session.add(step)
    session.commit()
    session.close()


teacher = TeacherQuestioner()
critic = CriticJudge()
student = MarginAnalyzer()

program = MarsAnalysisProgram(teacher, critic, student)

results = program(
    context=prompt,
    base_question="Is the company improving its profitability?"
)

dspy.inspect_history()
print(results)
