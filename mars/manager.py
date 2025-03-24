from openagents.agent import Agent

class ManagerAgent(Agent):
    async def run(self, context):
        print("Manager coordinating the agents")
        # Call UserProxy → Planner → TCS loop → Target
