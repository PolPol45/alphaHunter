import subprocess
import time
import sys
import os

print("🚀 Avviando l'Ecosistema a Micro-servizi (Test Mode)...")

# Passa PYTHONPATH per esporre la directory root a tutti i micro-servizi
env = os.environ.copy()
env["PYTHONPATH"] = os.getcwd()

# 1. Avvia il Broker in background (Sistema nervoso)
broker = subprocess.Popen([sys.executable, "infrastructure/zmq_broker.py"], env=env)
time.sleep(2) # lascia che apra le porte 5555 e 5556

# 2. Avvia i Consumer che si mettono in ascolto passivo
ml = subprocess.Popen([sys.executable, "services/ml_consumer.py"], env=env)
exec = subprocess.Popen([sys.executable, "services/execution_consumer.py"], env=env)
time.sleep(2)

print("\n📡 Tutto pronto! Le reti sono in ascolto. Lancio il feed Dati (Il 'La' che scatena la catena...)")

# 3. Mettiamo un override volante nel data_producer per farlo girare una volta sola e non all'infinito
with open("services/data_producer.py", "r") as f:
    codice = f.read()

codice_test = codice.replace("while True:", "for _ in range(1):")

with open("services/data_producer_test.py", "w") as f:
    f.write(codice_test)

# Lanciamo il producer che fa 1 ciclo e si spegne
prod = subprocess.Popen([sys.executable, "services/data_producer_test.py"], env=env)

print("⏳ Attendo la fine della simulazione a reazione a catena (Il Producer scaricher\u00e0 i file parquet Polars...)")
try:
    prod.wait() # Aspetta che il fetch dati finisca
    time.sleep(30) # D\u00e0 abbondantemente tempo a ML e Exec di ricevere l'evento e processarlo
finally:
    print("\n🛑 Test finito! Abbatto i micro-servizi asincroni (spegnimento controllato)...")
    ml.terminate()
    exec.terminate()
    broker.terminate()
    print("✅ Tutte le macchine arrestate con successo!")
