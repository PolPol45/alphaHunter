# Trading Bot Momentum — Piano Strategico per Account €1.000

**Autore:** Claude per Paolo
**Data:** 6 Aprile 2026
**Obiettivo:** Costruire un bot Python per day trading intraday su crypto + mercati tradizionali
**Capitale iniziale:** €1.000

> ⚠️ **Disclaimer:** Il trading algoritmico comporta rischi significativi. Questo documento è un piano tecnico-strategico, non consulenza finanziaria. Potresti perdere parte o tutto il capitale investito.

---

## 1. SCELTA DEL BROKER — Analisi Comparativa

### 1.1 Crypto: Binance vs Kraken

| Caratteristica | Binance | Kraken |
|---|---|---|
| **Fee Spot (VIP 0)** | Maker 0.10%, Taker 0.10% | Maker 0.25%, Taker 0.40% |
| **Sconto BNB/KFEE** | -25% con BNB → 0.075% | Sconti volume (non rilevanti per €1k) |
| **Fee Futures** | Maker 0.02%, Taker 0.05% | Maker 0.02%, Taker 0.05% |
| **Min. ordine** | ~$10 (varia per coppia) | ~$10 (varia per coppia) |
| **API Rate Limit** | 1.200 req/min (weight-based) | 15-20 req/sec (tier-based) |
| **Libreria Python** | `ccxt`, `python-binance` (7.1k ⭐) | `ccxt`, `krakenex` |
| **WebSocket** | Sì, streams multipli | Sì, ma meno documentato |
| **Testnet/Sandbox** | Sì (testnet.binance.vision) | Sì (demo.kraken.com) |
| **Residenti EU** | Attenzione: Binance ha restrizioni in alcuni paesi EU | Piena operatività EU, regolamentato |

**Verdetto Crypto:** Per un conto da €1.000, **Binance** ha fee nettamente inferiori (soprattutto con sconto BNB). Tuttavia, **verifica la disponibilità per residenti italiani** — Binance ha avuto restrizioni CONSOB. Kraken è l'alternativa più sicura dal punto di vista regolamentare EU.

**Impatto fee su €1.000 (10 trade/giorno, position media €200):**
- Binance con BNB: ~€3/giorno → ~€60/mese
- Kraken spot: ~€13/giorno → ~€260/mese (insostenibile!)
- Kraken Pro: ~€5.20/giorno → ~€104/mese

### 1.2 Mercati Tradizionali: Interactive Brokers vs eToro

| Caratteristica | Interactive Brokers (IBKR) | eToro |
|---|---|---|
| **API Pubblica** | ✅ TWS API, IB Gateway, Client Portal | ❌ **Nessuna API pubblica per trading** |
| **Commissioni Azioni EU** | 0.05% (min €1.25) | 0% (spread incluso) |
| **Commissioni Azioni US** | $0.005/share (min $1) | 0% (spread incluso) |
| **Forex** | 0.08-0.2 bps sul volume | Spread 1-3 pip |
| **Min. Deposito** | Nessuno (era $10k, rimosso) | $50 |
| **Libreria Python** | `ib_insync` (molto matura) | Nessuna ufficiale |
| **Adatto a bot?** | ✅ Eccellente | ❌ No — solo manuale/copy trading |

**Verdetto Tradizionale:** **Interactive Brokers è l'unica scelta** per trading algoritmico su azioni/ETF/forex. eToro non ha API pubblica per bot — eliminalo dalle opzioni.

### 1.3 Configurazione Consigliata

```
CRYPTO (70% del capitale = €700)
├── Broker: Binance (o Kraken se Binance non disponibile in IT)
├── Libreria: ccxt + python-binance
├── Coppie: BTC/USDT, ETH/USDT, SOL/USDT (alta liquidità)
└── Fee effettive: ~0.075% con BNB

TRADIZIONALE (30% del capitale = €300)
├── Broker: Interactive Brokers (IBKR Pro)
├── Libreria: ib_insync
├── Asset: SPY, QQQ, top azioni momentum
└── Fee: ~$1-2 per trade
```

> **Nota critica:** Con €300 su IBKR, le commissioni minime (€1.25/trade) pesano ~0.4% per operazione. Valuta se il mercato tradizionale ha senso con questo capitale — potrebbe essere meglio concentrare tutto su crypto finché il conto non cresce.

---

## 2. FRAMEWORK E STACK TECNOLOGICO

### 2.1 Confronto Framework Open-Source

| Framework | ⭐ GitHub | Crypto | Azioni | Forex | Momentum | Difficoltà |
|---|---|---|---|---|---|---|
| **Freqtrade** | 48.400 | ✅ | ❌ | ❌ | ✅ | Media |
| **Jesse** | 7.600 | ✅ | ❌ | ❌ | ✅ | Media |
| **Blankly** | 2.400 | ✅ | ✅ | ✅ | ✅ | Bassa |
| **Backtrader** | 21.000 | ⚠️ | ✅ | ✅ | ✅ | Alta |
| **CCXT** (libreria) | 41.700 | ✅ | ❌ | ❌ | N/A | Media |
| **Backtesting.py** | 8.200 | ✅ | ✅ | ✅ | ✅ | Bassa |
| **VectorBT** | 4.600 | ✅ | ✅ | ✅ | ✅ | Media-Alta |

### 2.2 Stack Raccomandato

**Opzione A — Massima semplicità (consigliata per iniziare):**

```
BACKTESTING & RICERCA          LIVE TRADING
├── Backtesting.py             ├── Freqtrade (crypto)
├── VectorBT (analisi veloce)  ├── ib_insync (azioni, futuro)
├── pandas + ta-lib             └── ccxt (connettore exchange)
└── yfinance (dati storici)
```

**Opzione B — Framework unico multi-asset:**

```
Blankly
├── Crypto: Binance, Coinbase, KuCoin
├── Azioni: Alpaca (US) — nota: non copre EU
├── Forex: OANDA
└── Backtesting integrato con fee simulation
```

**Opzione C — Massimo controllo (bot custom):**

```
Bot Python personalizzato
├── ccxt → connessione exchange crypto
├── ib_insync → connessione IBKR
├── ta (o ta-lib) → indicatori tecnici
├── pandas → manipolazione dati
├── schedule / APScheduler → timing
├── SQLite → logging trade
└── python-telegram-bot → notifiche
```

### 2.3 Raccomandazione

**Per il tuo profilo (Python intermedio+, €1.000, day trading):**

1. **Fase 1 (mesi 1-2):** Usa **Freqtrade** per crypto. Ha il miglior ecosistema, backtesting robusto, dry-run mode, e gestione fee integrata. Comunità enorme.
2. **Fase 2 (mese 3+):** Quando il conto cresce, aggiungi **ib_insync** per mercati tradizionali come modulo separato.
3. **Backtesting:** Usa **Backtesting.py** o il backtest integrato di Freqtrade per validare ogni strategia.

---

## 3. STRATEGIE DI MOMENTUM

### 3.1 Indicatori Chiave per Momentum Intraday

**Indicatori Primari (segnali di entrata):**

- **RSI (Relative Strength Index, periodo 14):** Ipercomprato >70, ipervenduto <30. Per intraday usa anche RSI a 7 periodi per segnali più veloci.
- **MACD (12, 26, 9):** Crossover della signal line = segnale di momentum. Divergenza MACD/prezzo = inversione potenziale.
- **EMA Crossover (9/21):** EMA 9 sopra EMA 21 = momentum rialzista. Veloce e affidabile su timeframe 5m-15m.

**Indicatori di Conferma:**

- **ADX (Average Directional Index, 14):** Sopra 25 = trend forte (conferma momentum). Sotto 20 = mercato laterale (stai fuori).
- **Volume Profile / OBV:** Volume crescente conferma il movimento. Senza volume, il segnale è debole.
- **VWAP (Volume Weighted Average Price):** Prezzo sopra VWAP = bias rialzista intraday. Fondamentale per day trading.
- **Bollinger Bands (20, 2):** Breakout oltre la banda con volume = momentum entry. Squeeze = esplosione imminente.

### 3.2 Strategia Combinata Consigliata: "Triple Momentum Filter"

Questa strategia usa 3 filtri indipendenti per ridurre i falsi segnali:

```
SEGNALE LONG (tutti e 3 devono essere veri):
├── 1. EMA 9 > EMA 21 (trend direction)
├── 2. RSI(14) tra 40-70 (momentum senza ipercomprato)
├── 3. ADX > 25 (trend forte)
└── CONFERMA: Volume > media 20 periodi

SEGNALE SHORT (tutti e 3 devono essere veri):
├── 1. EMA 9 < EMA 21
├── 2. RSI(14) tra 30-60
├── 3. ADX > 25
└── CONFERMA: Volume > media 20 periodi

TIMEFRAME: 15 minuti (primario) + 1 ora (filtro trend)
```

### 3.3 Strategia Alternativa: "Bollinger Momentum Breakout"

```
ENTRATA LONG:
├── Prezzo chiude sopra Bollinger Band superiore
├── RSI(7) > 50 e in salita
├── Volume > 1.5x media 20 periodi
└── VWAP: prezzo sopra VWAP

ENTRATA SHORT:
├── Prezzo chiude sotto Bollinger Band inferiore
├── RSI(7) < 50 e in discesa
├── Volume > 1.5x media 20 periodi
└── VWAP: prezzo sotto VWAP

USCITA:
├── Target: ritorno alla media BB (linea centrale)
├── Stop-loss: 1.5x ATR(14)
└── Trailing stop: attivato dopo +1% di profitto
```

### 3.4 Strategia per Crypto Specifica: "MACD Divergence + Volume"

```
SETUP:
├── Timeframe: 5 minuti
├── Coppie: BTC/USDT, ETH/USDT
└── Orari migliori: 14:00-22:00 CET (overlap US/EU)

SEGNALE:
├── Divergenza bullish MACD (prezzo fa nuovo minimo, MACD no)
├── RSI(14) < 35 (zona di interesse)
├── Volume spike > 2x media
└── Prezzo sopra EMA 200 (trend di fondo rialzista)

GESTIONE:
├── Entry: al close della candela di conferma
├── Stop-loss: sotto il minimo recente (-1% max)
├── Take-profit 1: +1.5% (chiudi 50%)
└── Take-profit 2: +3% (chiudi il resto con trailing)
```

---

## 4. GESTIONE DEL RISCHIO — Cruciale per €1.000

### 4.1 Regole Ferree

| Regola | Valore | Motivazione |
|---|---|---|
| **Max rischio per trade** | 1-2% del capitale (€10-20) | Una serie di 5 loss = -10%, recuperabile |
| **Max trade aperti** | 2-3 contemporanei | Diversificazione senza sovraesposizione |
| **Max perdita giornaliera** | 3% (€30) | Stop trading per la giornata se raggiunto |
| **Max perdita settimanale** | 6% (€60) | Stop bot per la settimana se raggiunto |
| **Risk/Reward minimo** | 1:2 | Non entrare se il target < 2x lo stop |
| **Max % capitale per trade** | 20% (€200) | Non mettere tutto su un singolo trade |

### 4.2 Position Sizing — Formula

```python
def calculate_position_size(capital, risk_pct, entry_price, stop_loss_price):
    """
    Calcola la size della posizione basata sul rischio.

    capital: 1000 (EUR)
    risk_pct: 0.01 (1%)
    entry_price: es. 65000 (BTC)
    stop_loss_price: es. 64350 (BTC)
    """
    risk_amount = capital * risk_pct  # €10
    price_risk = abs(entry_price - stop_loss_price)  # 650
    price_risk_pct = price_risk / entry_price  # 1%
    position_size = risk_amount / price_risk_pct  # €1000

    # Ma limitato al max 20% del capitale
    max_position = capital * 0.20  # €200
    return min(position_size, max_position)
```

### 4.3 Kelly Criterion (semplificato)

```
f = W - (1-W)/R

dove:
  W = win rate della strategia (dal backtest)
  R = rapporto avg_win / avg_loss
  f = frazione ottimale del capitale per trade

Esempio:
  W = 55%, R = 2.0
  f = 0.55 - (0.45/2.0) = 0.55 - 0.225 = 0.325 (32.5%)

  → Usa f/2 (half-Kelly) = 16.25% per sicurezza
```

### 4.4 Impatto Fee — Simulazione Realistica

```
SCENARIO: 10 trade/giorno, position media €200, Binance con BNB

Fee per trade (round trip):
  Entry: €200 × 0.075% = €0.15
  Exit:  €200 × 0.075% = €0.15
  Totale per trade: €0.30

Fee giornaliere: 10 × €0.30 = €3.00
Fee mensili (22 giorni): €66.00

→ Devi generare ALMENO +6.6% al mese solo per coprire le fee
→ Questo è il motivo per cui il position sizing e la qualità dei segnali
  sono più importanti della quantità di trade
```

**Ottimizzazione fee:**

- Riduci il numero di trade: meglio 3-5 trade di qualità che 15 mediocri
- Usa ordini limit (maker) invece di market (taker) dove possibile — fee più basse
- Su Binance: compra BNB e attiva il pagamento fee in BNB (-25%)
- Evita coppie con bassa liquidità (spread più ampio = costo nascosto)

---

## 5. ARCHITETTURA DEL BOT

### 5.1 Schema ad Alto Livello

```
┌─────────────────────────────────────────────────┐
│                   TRADING BOT                     │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  DATA     │  │ STRATEGY │  │  EXECUTION   │   │
│  │  ENGINE   │→ │  ENGINE  │→ │  ENGINE      │   │
│  │          │  │          │  │              │   │
│  │ - OHLCV  │  │ - RSI    │  │ - Order mgmt │   │
│  │ - Volume │  │ - MACD   │  │ - Position   │   │
│  │ - Book   │  │ - EMA    │  │ - Stop/TP    │   │
│  │ - News   │  │ - ADX    │  │ - Fee calc   │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
│       ↑                            ↓              │
│  ┌──────────┐              ┌──────────────┐      │
│  │ EXCHANGE │              │   RISK        │      │
│  │ ccxt /   │              │   MANAGER     │      │
│  │ ib_insync│              │              │      │
│  └──────────┘              │ - Max loss    │      │
│       ↑                    │ - Position sz │      │
│  ┌──────────┐              │ - Daily limit │      │
│  │ WebSocket│              └──────────────┘      │
│  │ Stream   │                    ↓               │
│  └──────────┘              ┌──────────────┐      │
│                            │  LOGGER /     │      │
│                            │  TELEGRAM     │      │
│                            └──────────────┘      │
└─────────────────────────────────────────────────┘
```

### 5.2 Moduli Python

```
trading_bot/
├── config/
│   ├── settings.py          # API keys, parametri globali
│   ├── pairs.py             # Coppie da tradare
│   └── risk_params.py       # Parametri di rischio
├── data/
│   ├── fetcher.py           # Scarica OHLCV via ccxt/websocket
│   ├── indicators.py        # Calcolo indicatori (ta-lib/ta)
│   └── market_state.py      # Stato corrente del mercato
├── strategy/
│   ├── base_strategy.py     # Classe base astratta
│   ├── triple_momentum.py   # Strategia EMA+RSI+ADX
│   ├── bollinger_breakout.py # Strategia BB momentum
│   └── macd_divergence.py   # Strategia divergenza MACD
├── execution/
│   ├── order_manager.py     # Piazzamento e gestione ordini
│   ├── position_tracker.py  # Tracking posizioni aperte
│   └── fee_calculator.py    # Calcolo fee in tempo reale
├── risk/
│   ├── risk_manager.py      # Regole di rischio
│   ├── position_sizer.py    # Calcolo size posizioni
│   └── daily_limits.py      # Limiti giornalieri/settimanali
├── backtest/
│   ├── backtester.py        # Engine di backtesting
│   ├── optimizer.py         # Ottimizzazione parametri
│   └── report.py            # Report performance
├── notifications/
│   ├── telegram_bot.py      # Notifiche Telegram
│   └── logger.py            # Logging su file + DB
├── db/
│   └── models.py            # SQLite: trade, performance, log
├── main.py                  # Entry point
└── requirements.txt
```

### 5.3 Dipendenze Principali

```txt
# requirements.txt
ccxt>=4.0                    # Connessione exchange crypto
ib_insync>=0.9               # Connessione Interactive Brokers
ta>=0.11                     # Indicatori tecnici (puro Python)
# oppure: TA-Lib             # Più veloce ma richiede C library
pandas>=2.0                  # Manipolazione dati
numpy>=1.24                  # Calcoli numerici
python-telegram-bot>=20.0    # Notifiche
APScheduler>=3.10            # Scheduling
backtesting>=0.3             # Backtesting
vectorbt>=0.26               # Analisi performance
yfinance>=0.2                # Dati storici azioni
sqlalchemy>=2.0              # ORM per database
loguru>=0.7                  # Logging avanzato
```

---

## 6. PIANO DI IMPLEMENTAZIONE

### Fase 1 — Setup e Backtesting (Settimane 1-2)

```
□ Installa Freqtrade (Docker consigliato)
□ Configura connessione Binance testnet
□ Implementa strategia "Triple Momentum Filter"
□ Scarica dati storici (almeno 6 mesi, candele 15m)
□ Esegui backtest con fee realistiche (0.075%)
□ Analizza risultati: win rate, max drawdown, Sharpe ratio
□ Target minimo: win rate >50%, profit factor >1.5
```

### Fase 2 — Paper Trading (Settimane 3-4)

```
□ Attiva dry-run mode su Freqtrade
□ Monitora per 2 settimane senza soldi reali
□ Confronta risultati dry-run vs backtest
□ Ottimizza parametri se necessario
□ Imposta notifiche Telegram
□ Documenta ogni trade e il motivo
```

### Fase 3 — Live Trading Micro (Settimane 5-6)

```
□ Deposita €100-200 su Binance (NON tutto il capitale!)
□ Attiva bot con position size ridotta (€50/trade)
□ Monitora attentamente per 2 settimane
□ Verifica che le fee corrispondano alle aspettative
□ Verifica slippage reale vs simulato
```

### Fase 4 — Scale Up (Mese 2-3)

```
□ Se i risultati sono positivi, aumenta gradualmente
□ Aggiungi seconda strategia (Bollinger Breakout)
□ Considera l'aggiunta di IBKR per azioni
□ Implementa reporting automatico settimanale
□ Target: tutto il capitale €1.000 attivo
```

---

## 7. RISORSE E REPOSITORY GITHUB ESSENZIALI

### Repository da Studiare

| Repository | ⭐ | Cosa Impari |
|---|---|---|
| `freqtrade/freqtrade` | 48.4k | Framework completo, strategie, backtesting |
| `ccxt/ccxt` | 41.7k | Connessione a 110+ exchange |
| `mementum/backtrader` | 21k | Architettura event-driven trading |
| `microsoft/qlib` | 40.3k | ML per finanza quantitativa |
| `jesse-ai/jesse` | 7.6k | Strategie eleganti, 300+ indicatori |
| `kernc/backtesting.py` | 8.2k | Backtesting leggero e veloce |
| `polakowo/vectorbt` | 4.6k | Analisi vettoriale ultra-veloce |
| `blankly-finance/blankly` | 2.4k | Multi-asset (crypto+azioni+forex) |
| `sammchardy/python-binance` | 7.1k | API Binance completa |
| `erdewit/ib_insync` | 2.8k | API Interactive Brokers Pythonic |

### Risorse di Apprendimento

**Community:**
- `r/algotrading` — Subreddit principale, molto attivo
- `r/cryptocurrency` — Analisi mercato crypto
- Freqtrade Discord — Supporto diretto dal team
- QuantConnect Community — Strategie quantitative

**YouTube (canali consigliati per algotrading Python):**
- **Part Time Larry** — Tutorial ccxt, Alpaca, bot crypto
- **Algovibes** — Strategie Python con backtesting
- **The Trading Parrot** — Freqtrade specifico
- **Coding Jesus** — Bot trading da zero

**Documentazione:**
- docs.freqtrade.io — Guida completa Freqtrade
- docs.ccxt.com — Manuale CCXT
- interactivebrokers.github.io — API IBKR
- ta-lib.github.io/ta-lib-python — Indicatori tecnici

---

## 8. ERRORI COMUNI DA EVITARE

1. **Overfitting nel backtest:** Una strategia che fa +500% nel backtest probabilmente non funzionerà live. Usa walk-forward analysis e out-of-sample testing.

2. **Ignorare le fee:** Con €1.000, le fee sono il tuo nemico numero uno. Ogni centesimo conta. Simula SEMPRE con fee realistiche.

3. **Overtrading:** Più trade ≠ più profitto. Meno trade di qualità > molti trade mediocri. Target: 3-5 trade/giorno max.

4. **Nessun risk management:** Il bot DEVE avere stop-loss automatici. Mai "sperare" che il mercato torni a tuo favore.

5. **Passare da backtest a live troppo velocemente:** SEMPRE fare almeno 2 settimane di paper trading.

6. **Usare tutto il capitale subito:** Inizia con €100-200. Scala solo dopo risultati verificati.

7. **Ignorare la liquidità:** Su crypto minori, lo slippage può mangiarsi tutto il profitto.

8. **Non loggare:** Ogni trade deve essere registrato con motivo di entrata, uscita, e risultato. Senza dati non puoi migliorare.

---

## 9. ASPETTATIVE REALISTICHE

Con un conto da €1.000, un bot ben costruito e ottimizzato potrebbe realisticamente generare:

| Scenario | Rendimento Mensile | Dopo 12 Mesi |
|---|---|---|
| **Conservativo** | +2-3% | €1.268 - €1.426 |
| **Moderato** | +5-8% | €1.796 - €2.518 |
| **Aggressivo** | +10-15% | €3.138 - €5.350 |
| **Irrealistico** ⚠️ | +30%+ | Probabile perdita totale |

> Chi promette rendimenti del 30%+ mensile costanti sta mentendo. I migliori hedge fund quant al mondo fanno 15-25% **annuo**. Un bot retail competitivo punta al 5-10% mensile nei periodi buoni, con mesi negativi inevitabili.

---

## 10. CHECKLIST RAPIDA DI AVVIO

```
✅ Passo 1: Crea account Binance (o Kraken) + genera API key (solo trading, NO withdrawal)
✅ Passo 2: Installa Freqtrade → docker-compose up
✅ Passo 3: Configura connessione testnet Binance
✅ Passo 4: Implementa strategia Triple Momentum
✅ Passo 5: Backtest su 6 mesi di dati (BTC/USDT 15m)
✅ Passo 6: Se profit factor > 1.5 → dry-run 2 settimane
✅ Passo 7: Se dry-run positivo → live con €100
✅ Passo 8: Scala gradualmente fino a €1.000
✅ Passo 9: Aggiungi strategie e asset progressivamente
✅ Passo 10: Review settimanale dei risultati
```

---

*Report generato il 6 Aprile 2026. Le fee e condizioni dei broker potrebbero essere cambiate — verifica sempre sui siti ufficiali prima di operare.*
