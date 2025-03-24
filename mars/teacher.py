from openagents.agent import Agent

class TeacherAgent(Agent):
    async def run(self, context):
        step = context['current_step']
        prev_prompt = context['current_prompt']
        question = f"What is missing in this prompt to improve step: {step}?"
        return {"question": question}
