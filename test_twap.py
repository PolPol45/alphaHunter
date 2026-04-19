import logging
from agents.execution_agent import ExecutionAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")
agent = ExecutionAgent()
print("Starting Execution Agent in Live Paper Mode...")
agent.run()
print("Done!")
