import logging
from infrastructure.zmq_pubsub import EventBus
from agents.execution_agent import ExecutionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExecutionConsumer")

def run():
    bus_sub = EventBus("SUB")
    bus_sub.subscribe("SIGNALS_READY")
    
    exec_agt = ExecutionAgent()
    
    logger.info("🔫 Execution Consumer armato. Attende passivamente ordini sul Bus...")
    
    while True:
        topic, payload = bus_sub.receive()
        logger.info(f"Ricevuto evento {topic}. I segnali hanno superato risk, avvio il motore TWAP/Execution!")
        
        try:
            exec_agt.run()
            logger.info("Esecuzione ordini completata con successo.")
        except Exception as e:
            logger.error(f"Execution Error: {e}")

if __name__ == "__main__":
    run()
