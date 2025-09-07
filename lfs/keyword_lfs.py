# -*- coding: utf-8 -*-
import re
from typing import Optional, Dict, List

LABELS = ["KIS","How-to","Music","News","Sports","Review","Entertainment","Other"]

# Compiled regex per class
RX = {
    "Music": re.compile(r"\blyrics?\b|karaoke|\bmv\b|official\s+mv|official\s+audio|\blofi(\s+chill)?\b|\bnhạc\b|\binstrumental\b|\bremix\b|\bplaylist\b", re.I),
    "How-to": re.compile(r"\bhow\s*to\b|tutorial|hướng\s*dẫn|\bcách\b|\bfix\b|\bsửa\b|\bmẹo\b|guide|step\s*by\s*step|install|setup|cài\s*đặt|khắc\s*phục", re.I),
    "Sports": re.compile(r"\b(highlight|highlights)\b|\bfull\s*match\b|\bvs\b|\btrận\b|bán\s*kết|chung\s*kết|world\s*cup|premier\s*league|\blive\b|\bgoals?\b|\bscore\b", re.I),
    "News": re.compile(r"\bnews\b|breaking|thời\s*sự|bản\s*tin|\btin\s*tức\b|\bnóng\b|evening\s*news|morning\s*news", re.I),
    "Review": re.compile(r"\breview\b|đánh\s*giá|so\s*sánh|\bunboxing\b|\bgiá\b|\bcompare\b|\bhands?-?on\b|\btrên\s*tay\b", re.I),
    "KIS": re.compile(r"trailer|official\s*trailer|episode|\bep\.?\b|season|\btập\s*\d+\b|vietsub|full\s*movie|\bchapter\b|\bchương\b|\bphần\s*\d+\b", re.I),
    "Entertainment": re.compile(r"\bvlog\b|reaction|pranks?|challenge|funny|memes?|skit|talk\s*show|podcast|live\s*stream|livestream|streaming|asmr|parody", re.I),
    # "Other" will be used as fallback; we don't force positive regex for it
}

# Helpers for tie-breaking
RX_SO_SANH_VS = re.compile(r"so\s*sánh.*\bvs\b", re.I)
RX_EP_INDICATORS = re.compile(r"\btập\s*\d+|\bepisode\b|\bep\.?\b|\bseason\b|\bphần\s*\d+|\bchapter\b|\bchương\b|\bvietsub\b", re.I)
RX_MUSIC_STRONG = re.compile(r"\blyrics?\b|\bkaraoke\b|\bofficial\s+mv\b|\bmv\b|\bofficial\s+audio\b", re.I)
RX_SPORTS_STRONG = re.compile(r"\b(highlight|highlights|full\s*match|goal|goals|world\s*cup|premier\s*league|champions\s*league|live\b)", re.I)

def votes_dict(q: str) -> Dict[str, int]:
    votes = {lab: 0 for lab in LABELS}
    if not isinstance(q, str): 
        return votes
    for lab, rx in RX.items():
        if rx.search(q):
            votes[lab] = 1
    return votes

def _tie_break(q: str, matched: List[str]) -> Optional[str]:
    t = q.lower()

    # 1) Review vs Sports
    if "Sports" in matched and "Review" in matched:
        if RX_SO_SANH_VS.search(t):
            matched = [m for m in matched if m != "Sports"]
        elif RX_SPORTS_STRONG.search(t):
            matched = [m for m in matched if m != "Review"]

    # 2) Music vs KIS
    if "Music" in matched and "KIS" in matched:
        if RX_EP_INDICATORS.search(t) and not RX_MUSIC_STRONG.search(t):
            matched = [m for m in matched if m != "Music"]
        elif RX_MUSIC_STRONG.search(t):
            matched = [m for m in matched if m != "KIS"]

    # 3) News có pattern rõ ràng => ưu tiên News
    if "News" in matched and re.search(r"\bnews\b|thời\s*sự|bản\s*tin|tin\s*tức", t, re.I):
        return "News"

    # 4) Nếu còn nhiều nhãn đặc thù + "Entertainment", ưu tiên nhãn đặc thù
    if "Entertainment" in matched and len(matched) > 1:
        spec = [m for m in matched if m != "Entertainment"]
        if len(spec) == 1:
            return spec[0]

    # 5) Nếu còn đúng 1 nhãn
    if len(matched) == 1:
        return matched[0]

    return None

def predict_rules_only(q: str, *, include_other: bool = True) -> str:
    """
    Return one of 8 labels. If ambiguous or no match:
      - include_other=True (default): return "Other"
      - include_other=False: return "" (blank)
    """
    v = votes_dict(q)
    matched = [lab for lab, c in v.items() if c > 0]

    if len(matched) == 0:
        return "Other" if include_other else ""

    if len(matched) == 1:
        return matched[0]

    tb = _tie_break(q, matched)
    if tb is not None:
        return tb

    # Still ambiguous
    return "Other" if include_other else ""
