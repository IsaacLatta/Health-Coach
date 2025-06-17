import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any

_DATA_SET_PATH_ =  Path(__file__).resolve().parents[1]/"resources"/"data"/"Heart_Disease_Prediction.csv"

def extract_anchor_pairs(csv_path: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    n = len(records)
    even_count = n if (n % 2 == 0) else (n - 1)
    if even_count < 2:
        return [] 

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i in range(0, even_count, 2):
        pairs.append((records[i], records[i+1]))

    return pairs

def main():
    print(f"Extract anchor pairs from data set: \"{_DATA_SET_PATH_}\" ...")
    anchors = extract_anchor_pairs(_DATA_SET_PATH_)
    print(f"Found {len(anchors)} anchor pairs.")

if __name__ == "__main__":
    main()