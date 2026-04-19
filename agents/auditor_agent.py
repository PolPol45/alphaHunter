import json
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

from agents.base_agent import BaseAgent, DATA_DIR

class AuditorAgent(BaseAgent):
    """
    Fase 14: Daily Auditor Agent.
    Interroga i log di chiusura di fine giornata, ne estrae i PnL e usa un LLM
    per stilare una narrazione del comportamento del bot (Cosa è successo, perché abbiamo guadagnato/perso).
    """
    def __init__(self):
        super().__init__("auditor_agent")
        
    def run(self) -> bool:
        self.mark_running()
        try:
            self.logger.info("Avviando Nightly Audit LLM...")
            llm_config = self.config.get("llm_nlp", {})
            if not llm_config.get("enabled", False):
                self.logger.info("LLM Auditor disabled via config. Skipping.")
                self.mark_done()
                return True
                
            # Recupera trade passati / performance (mocking se i path non sono attivi)
            portfolio_file = DATA_DIR / "simulated_portfolio.json"
            if not portfolio_file.exists():
                self.logger.info("Nessun portafoglio simulato da analizzare.")
                self.mark_done()
                return True
                
            portf_data = self.read_json(portfolio_file)
            pnl = portf_data.get("pnl_percentage", "0.0")
            cash = portf_data.get("cash", "0.0")
            
            # Recuperiamo un estratto del macro regime odierno
            macro_file = DATA_DIR / "market_regime.json"
            macro_data = self.read_json(macro_file) or {}
            regime = macro_data.get("regime", "UNKNOWN")
            
            prompt = (
                f"Sei il Chief Risk Officer del fondo. Oggi il mercato era {regime}.\\n"
                f"Il portafoglio ha generato un PnL temporaneo del {pnl}%, chiudendo con cash {cash}.\\n"
                "Scrivi un breve summary confidenziale di 3 righe da mandare su Telegram al PM per giustificare questa chiusura."
            )
            
            provider = llm_config.get("provider", "openrouter")
            api_key = llm_config.get("api_key", "")
            model = llm_config.get("model", "meta-llama/llama-3-8b-instruct:free")
            
            narrative = "Audit non disponibile."
            
            if provider == "openrouter" and api_key:
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Sei un reporter finanziario acuto e sintetico."},
                        {"role": "user", "content": prompt}
                    ]
                }
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    narrative = resp.json()['choices'][0]['message']['content']
            elif provider == "ollama":
                endpoint = llm_config.get("endpoint", "http://localhost:11434/api/generate")
                resp = requests.post(endpoint, json={"model": model, "prompt": prompt, "stream": False}, timeout=30)
                if resp.status_code == 200:
                    narrative = resp.json().get("response", "")
                    
            self.logger.info(f"\\n--- DAILY AUDIT REPORT ---\\n{narrative}\\n-------------------------")
            
            # Save the narrative
            audit_report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": pnl,
                "regime": regime,
                "llm_summary": narrative
            }
            self.write_json(DATA_DIR / "daily_audit_report.json", audit_report)
            
            self.mark_done()
            return True
            
        except Exception as e:
            self.logger.warning(f"Auditor fallito: {e}")
            self.mark_error(e)
            return False
