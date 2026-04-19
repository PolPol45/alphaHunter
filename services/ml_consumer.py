import time
import logging
from infrastructure.zmq_pubsub import EventBus
from agents.ml_strategy_agent import MLStrategyAgent
from agents.risk_agent import RiskAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLConsumer")

def run():
    bus_sub = EventBus("SUB")
    bus_pub = EventBus("PUB")
    
    # Questo modulo si sveglia SOLO quando il data producer grida "MARKET_DATA_READY"
    bus_sub.subscribe("MARKET_DATA_READY")
    
    ml_agt = MLStrategyAgent()
    risk_agt = RiskAgent()
    
    logger.info("🧠 ML Consumer in ascolto continuo del Bus ZMQ...")
    
    while True:
        topic, payload = bus_sub.receive()
        logger.info(f"Ricevuto evento {topic} -> Scateno la Pipeline ML!")
        
        try:
            ml_agt.run()
            # Valida subito il prop-desk risk
            risk_agt.run()
            
            # Se la pipe \u00e8 arrivata sana e salva fino in fondo, emette "SIGNALS_READY"
            bus_pub.publish("SIGNALS_READY", {"status": "validated_successfully", "ts": time.time()})
        except Exception as e:
            logger.error(f"ML Pipeline Failure durante processing evento: {e}")

if __name__ == "__main__":
    run()
