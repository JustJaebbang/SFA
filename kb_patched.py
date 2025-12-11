
import pandas as pd
import unicodedata
import re
from typing import Dict, Tuple, Optional

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def normalize_label(s: str) -> str:
    """
    Robust normalization for model labels and DB keys.
    - NFKC normalize
    - strip, lower (latin)
    - collapse whitespace → single underscore
    - replace hyphen→underscore
    - remove surrounding underscores
    """
    if s is None:
        return ""
    s = _nfkc(str(s))
    s = s.strip()
    # unify hyphen/whitespace
    s = s.replace("-", " ")
    s = re.sub(r"\s+", "_", s)
    # latin to lower; keep Korean as-is
    try:
        s = s.lower()
    except Exception:
        pass
    # remove duplicate underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def load_aliases(path: str) -> Dict[str, Tuple[str, str]]:
    """
    Returns: {alias_norm: (food_id, display_name_ko)}
    Accepts columns: alias, normalized(optional), food_id, display_name_ko(optional)
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    alias_col = cols.get("alias")
    food_col = cols.get("food_id")
    norm_col = cols.get("normalized")
    name_col = cols.get("display_name_ko")

    if alias_col is None or food_col is None:
        raise ValueError("aliases.csv must have columns: alias, food_id (normalized/display_name_ko optional)")

    idx: Dict[str, Tuple[str, str]] = {}
    for _, row in df.iterrows():
        food_id = str(row[food_col]).strip()
        disp = str(row[name_col]).strip() if name_col in row and pd.notna(row[name_col]) else ""
        # 1) alias normalized
        a1 = normalize_label(row[alias_col])
        if a1:
            idx[a1] = (food_id, disp)
        # 2) provided normalized column
        if norm_col:
            a2 = normalize_label(row[norm_col])
            if a2:
                idx[a2] = (food_id, disp)
        # 3) also register display name as alias (normalized)
        if disp:
            a3 = normalize_label(disp)
            if a3:
                idx[a3] = (food_id, disp)
    return idx

def _per100_from_row(row: pd.Series) -> Dict[str, float]:
    """
    Build per-100g dict. If basis != per_100g, rescale using serving_g.
    Expected nutrition columns: energy_kcal, carb_g, protein_g, fat_g, sodium_mg
    """
    # read values
    basis = str(row.get("basis", "per_100g"))
    serving_g = float(row.get("serving_g", 100) or 100)
    ek = float(row.get("energy_kcal", 0) or 0)
    cb = float(row.get("carb_g", 0) or 0)
    pr = float(row.get("protein_g", 0) or 0)
    ft = float(row.get("fat_g", 0) or 0)
    na = float(row.get("sodium_mg", 0) or 0)
    # if per serving (not 100g), scale to 100g
    if basis != "per_100g" and serving_g > 0:
        scale = 100.0 / serving_g
        ek, cb, pr, ft, na = ek*scale, cb*scale, pr*scale, ft*scale, na*scale
    per100 = {
        "kcal": round(ek, 3),
        "carb": round(cb, 3),
        "protein": round(pr, 3),
        "fat": round(ft, 3),
        "sodium_mg": round(na, 3),
        "sodium": round(na, 3),  # alias
    }
    return per100

def load_nutrition(nutrition_csv: str,
                   aliases_csv: Optional[str] = None,
                   include_display_keys: bool = True):
    """
    Returns:
      nutrition_map: Dict[str, Dict[str, float]]  # key → per-100g nutrition
      aliases_idx:   Dict[str, Tuple[str, str]]   # alias_norm → (food_id, display_name_ko)
    Keys inserted into nutrition_map:
      - food_id token without 'food_' prefix (e.g., 'food_fried_chicken' → 'fried_chicken')
      - if include_display_keys: normalized display_name_ko
      - if aliases_csv provided: all alias/normalized variants
    """
    ndf = pd.read_csv(nutrition_csv)
    cols = {c.lower(): c for c in ndf.columns}
    required = ["food_id", "display_name_ko", "basis", "serving_g",
                "energy_kcal", "carb_g", "protein_g", "fat_g", "sodium_mg"]
    for r in required:
        if r not in cols:
            raise ValueError(f"nutrition.csv missing column: {r}")

    food_col = cols["food_id"]
    name_col = cols["display_name_ko"]

    # build primary map by food_id token
    nutrition_map: Dict[str, Dict[str, float]] = {}
    token_to_food: Dict[str, str] = {}

    for _, row in ndf.iterrows():
        food_id = str(row[food_col]).strip()
        display = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
        # token from food_id (strip leading 'food_')
        token = food_id
        if token.startswith("food_"):
            token = token[5:]
        token_norm = normalize_label(token)
        per100 = _per100_from_row(row)
        nutrition_map[token_norm] = per100
        token_to_food[token_norm] = food_id

        if include_display_keys and display:
            key = normalize_label(display)
            nutrition_map[key] = per100

    # load aliases and add keys that point to the same per100 values
    if aliases_csv:
        alias_idx = load_aliases(aliases_csv)
        for a_norm, (food_id, disp) in alias_idx.items():
            # resolve to token key
            token = food_id
            if token.startswith("food_"):
                token = token[5:]
            token_norm = normalize_label(token)
            if token_norm in nutrition_map:
                nutrition_map[a_norm] = nutrition_map[token_norm]
    else:
        alias_idx = {}

    return nutrition_map, alias_idx

def get_nutrition_for_label(raw_label: str,
                            nutrition_map: Dict[str, Dict[str, float]],
                            aliases_idx: Optional[Dict[str, tuple]] = None):
    """
    Look up per-100g nutrition for a predicted label.
    1) normalize raw_label; direct lookup
    2) if not found and aliases_idx provided, try alias → food_id → token
    """
    key = normalize_label(raw_label)
    if key in nutrition_map:
        return nutrition_map[key]

    if aliases_idx:
        # alias to food_id
        if key in aliases_idx:
            food_id, _ = aliases_idx[key]
            token = food_id[5:] if str(food_id).startswith("food_") else str(food_id)
            token_norm = normalize_label(token)
            return nutrition_map.get(token_norm, None)
    return None

def scale_nutrition(per100: Dict[str, float], grams: float) -> Dict[str, float]:
    """
    Linear scaling from per-100g to grams.
    """
    scale = float(grams) / 100.0
    out = {}
    for k, v in per100.items():
        try:
            out[k] = round(float(v) * scale, 3)
        except Exception:
            pass
    return out
