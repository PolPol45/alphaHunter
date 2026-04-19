import time
import logging
from infrastructure.zmq_pubsub import EventBus
from agents.market_data_agent import MarketDataAgent
from agents.alternative_data_agent import AlternativeDataAgent
from agents.sentiment_analyzer_agent import SentimentAnalyzerAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataProducer")

def run():
    bus = EventBus("PUB")
    # Inizializza i vecchi agent trasformati in worker asincroni
    market = MarketDataAgent()
    sentiment = SentimentAnalyzerAgent()
    
    logger.info("📡 Data Producer avviato. In attesa di pompare feed (Ogni 60 sec)...")
    
    for _ in range(1):
        try:
            # 1. Tira gi\u00f9 i ticker
            market.run()
            # 2. Tira gi\u00f9 sentiment (meno di frequente)
            
            # Pubblica a tutti i nodi interessati che ci sono nuovi dati freschi
            bus.publish("MARKET_DATA_READY", {"status": "ok", "timestamp": time.time()})
            logger.info("Evento [MARKET_DATA_READY] trasmesso in rete.")
            
        except Exception as e:
            logger.error(f"Producer Crash: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    run()
