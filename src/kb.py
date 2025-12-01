import pandas as pd

def normalize_label(s:str):
    return s.strip().lower().replace("-","_").replace(" ","_")

def load_aliases(path:str):
    try:
        df = pd.read_csv(path)
        idx = {str(a).strip().lower().replace("-","_").replace(" ","_"): (fid, name)
               for a, fid, name in zip(df["alias"], df["food_id"], df.get("display_name_ko", [""]*len(df)))}
        return idx
    except Exception:
        return None

def load_nutrition(path:str):
    try:
        df = pd.read_csv(path).set_index("food_id")
        return df
    except Exception:
        return None
