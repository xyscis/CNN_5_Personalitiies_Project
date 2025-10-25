import pickle
from pathlib import Path

# Adjust root if necessary
root = Path("ChaLearn2016_tiny/annotation")

# All annotation files
files = [
    root / "annotation_training.pkl",
    root / "annotation_validation.pkl",
    root / "annotation_test.pkl",
]

def read_pkl(pkl_path):
    """Load and safely inspect pickle files."""
    print(f"\n--- Reading: {pkl_path.name} ---")
    try:
        with open(pkl_path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception:
                f.seek(0)
                data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"Error reading {pkl_path.name}: {e}")
        return

    # Overview
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())[:10]}")
        for key in list(data.keys())[:6]:
            val = data[key]
            print(f"  - {key}: type={type(val)}")
            if isinstance(val, dict):
                print(f"    inner keys sample: {list(val.keys())[:5]}")
                # print one example score
                if len(val) > 0:
                    first_vid = list(val.keys())[0]
                    print(f"    example: {first_vid} -> {val[first_vid]}")
    else:
        print("File does not contain a dictionary.")
    print("-" * 60)

for f in files:
    if f.exists():
        read_pkl(f)
    else:
        print(f"File not found: {f}")
