import requests
import json
import pandas as pd
import time
import sys

RPC_URL = "https://ethereum.publicnode.com"

def rpc_call(method, params=[]):
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    }
    response = requests.post(RPC_URL, json=payload)
    if response.status_code == 200:
        res = response.json()
        if 'error' in res:
            print(f"RPC Error: {res['error']}")
            return None
        return res.get("result")
    else:
        print(f"HTTP Error: {response.status_code}")
        return None

def fetch_live_transactions(target_count=1000):
    print(f"Connecting to {RPC_URL} to fetch latest block...")
    latest_block_hex = rpc_call("eth_blockNumber")
    if not latest_block_hex:
        print("Failed to get latest block number.")
        return
        
    latest_block_int = int(latest_block_hex, 16)
    print(f"Latest Block: {latest_block_int}")
    
    transactions = []
    current_block = latest_block_int
    
    while len(transactions) < target_count:
        print(f"Fetching Block {current_block}...")
        block_data = rpc_call("eth_getBlockByNumber", [hex(current_block), True])
        
        if not block_data:
            print("Failed to get block data. Retrying...")
            time.sleep(2)
            continue
            
        block_timestamp = int(block_data["timestamp"], 16)
        txs = block_data.get("transactions", [])
        tx_count = len(txs)
        
        for tx in txs:
            if len(transactions) >= target_count:
                break
                
            # Usually gasPrice is in Web3 transactions. EIP-1559 might use maxFeePerGas.
            gas_price_wei = int(tx.get("gasPrice", tx.get("maxFeePerGas", "0x0")), 16)
            gas_limit = int(tx.get("gas", "0x0"), 16)
            
            transactions.append({
                "hash": tx["hash"],
                "block_number": current_block,
                "timestamp": block_timestamp,
                "gas_price": gas_price_wei, # Wei
                "receipt_gas_used": gas_limit, # Proxying gas usage with gas limit to save millions of RPC calls
                "transaction_count": tx_count
            })
            
        sys.stdout.write(f"\rCollected: {len(transactions)} / {target_count}")
        sys.stdout.flush()
        current_block -= 1
        time.sleep(0.5) # Prevent rate limiting
        
    print("\nData collected successfully!")
    df = pd.DataFrame(transactions)
    
    # Save raw data
    raw_csv = "live_raw_transactions.csv"
    df.to_csv(raw_csv, index=False)
    print(f"Saved raw transactions to {raw_csv}")
    
    return raw_csv

if __name__ == "__main__":
    fetch_live_transactions(target_count=1000)
