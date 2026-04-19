import json

for file_path in ["data/portfolio_institutional.json", "data/portfolio_retail.json"]:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        trades = data.get("trades", [])
        print(f"=== {file_path} ===")
        print(f"Total trades recorded (Open+Close signals): {len(trades)}")
        
        sources = {}
        for t in trades:
            src = t.get("agent_source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            
        print(f"Agent Distribution (open+close): {sources}")
        
        open_pos = [p for p in data.get("positions", {}).values() if float(p.get("quantity", 0)) > 0]
        print(f"Currently open positions limit check: {len(open_pos)}")
        print(f"Current Equity Check: {data.get('total_equity')}")
    except Exception as e:
        print(f"Could not analyze {file_path}: {e}")
