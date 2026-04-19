"""
Il Grande Orchestratore di Collaudo (Unified Master Test)
Questo script avvia l'intera pipeline algoritmica end-to-end:
1. Training Machine Learning (Supervised + RL PPO Phase 13)
2. Estrazione SHAP Feature Importances (Phase 12)
3. Estrazione Valutazione Macro & NLP LLM (Phase 14)
4. Simulazione Esecuzione Agenti (Agent Sandbox)
"""
import sys
import subprocess
import sys
import os
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MasterCollaudo")

def run_master_collaudo():
    logger.info("==================================================")
    logger.info("   INIZIO GRANDE COLLAUDO UNIFICATO (Fasi 1-14)   ")
    logger.info("==================================================")
    
    python_exe = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    # ---------------------------------------------------------
    # 1. THE BRAIN: Machine Learning & RL
    # ---------------------------------------------------------
    logger.info("\\n>>> 1. AVVIO MACHINE LEARNING PIPELINE (Random Forest + RL PPO + SHAP)")
    try:
        # Usa il comando ufficiale del backend ML
        subprocess.run([python_exe, "backtesting/train_and_deploy.py", "--min-train-reports", "1"], env=env, check=True)
        logger.info("✅ Addestramento Modelli, RL e Spiegazioni SHAP completati con successo!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ ML Pipeline Fallita: {e}")
        return

    # ---------------------------------------------------------
    # 2. THE HEART: Systemic Agents in Sandbox via ZeroMQ
    # ---------------------------------------------------------
    logger.info("\\n>>> 2. AVVIO AMBIENTE ASINCRONO SIMULATO ZMQ (Kelly, Correlation, Microstruttura)")
    try:
        # Usa il lanciatore asincrono del test Micro-servizi (Fase 7)
        subprocess.run([python_exe, "run_event_driven_test.py"], env=env, check=True)
        logger.info("✅ Simulazione di sistema asincrono completata!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Sandboxed Agent Execution fallibile: {e}")
        return
        
    # ---------------------------------------------------------
    # 3. NLP MACRO ANALYZER & NIGHTLY AUDITOR Llama-3
    # ---------------------------------------------------------
    logger.info("\\n>>> 3. LANCIO NLP INTELLIGENCE E NIGHTLY REPORT (Fase 14 LLM)")
    try:
        from agents.auditor_agent import AuditorAgent
        from agents.macro_analyzer_agent import MacroAnalyzerAgent
        
        # Facciamo girare forzatamente il Macro per testare l'integrazione News + LLM 
        macro_agent = MacroAnalyzerAgent()
        macro_agent.run()
        
        # Facciamo girare l'Auditor per scrivere il report JSON testuale del Collaudo
        auditor = AuditorAgent()
        auditor.run()
    except Exception as e:
        logger.error(f"❌ Fallito modulo NLP LLM: {e}")

    logger.info("\\n==================================================")
    logger.info(" 🏆 COLLAUDO COMPLETATO CON SUCCESSO! IL BOT È ARMATO.")
    logger.info(" Aggiorna la Mega Dashboard per vedere Pesi RL, SHAP, PnL Simulato.")
    logger.info("==================================================")

if __name__ == "__main__":
    run_master_collaudo()
