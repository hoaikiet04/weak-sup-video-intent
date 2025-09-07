# -*- coding: utf-8 -*-
import re
from typing import Dict, List

LABELS = ["KIS","How-to","Music","News","Sports","Review","Entertainment","Other"]

PATTERNS = {
    "Music": r"\blyrics?\b|karaoke|\bmv\b|official\s+mv|official\s+audio|\blofi(\s+chill)?\b|\bnhạc\b|\b(remix|instrumental|playlist|soundtrack|ost)\b",
    "How-to": r"\bhow\s+to\b|tutorial|guide|instruction|hướng\s+dẫn|\bcách\b|\bfix\b|\bsửa\b|install|setup|troubleshoot|khắc\s+phục|cài\s+đặt",
    "Sports": r"\bvs\b|highlight(s)?|full\s*match|goal(s)?|match\b|\btrận\b|chung\s+kết|bán\s+kết|world\s+cup|premier\s+league|champions\s+league|\blive\b|score",
    "News": r"\bbreaking\s+news\b|\bnews\b|thời\s*sự|bản\s+tin|tin\s+tức|evening\s+news|morning\s+news|\bnóng\b",
    "Review": r"\breview\b|đánh\s+giá|so\s+sánh|\bunboxing\b|hands?-?on|trên\s*tay|\bgiá\b|\bcompare\b",
    "KIS": r"trailer|official\s*trailer|episode|\bep\.?\b|season|\btập\s*\d+\b|vietsub|full\s*movie|\bchapter\b|\bchương\b|\bphần\s*\d+\b",
    "Entertainment": r"\bvlog\b|reaction|pranks?|challenge|funny|memes?|skit|talk\s*show|podcast|live\s*stream|livestream|streaming|asmr|parody"
}

COMPILED = {k: re.compile(v, re.I) for k,v in PATTERNS.items()}

def keyword_hits(text: str) -> Dict[str, List[str]]:
    hits = {}
    for lab, rx in COMPILED.items():
        found = rx.findall(text or "")
        if found:
            vals = []
            for f in found:
                if isinstance(f, tuple):
                    vals.append(next((p for p in f if p), ""))
                else:
                    vals.append(f if isinstance(f,str) else str(f))
            hits[lab] = [v for v in vals if v]
    return hits

def rules_score(text: str) -> Dict[str, float]:
    hits = keyword_hits(text or "")
    votes = {lab: 0 for lab in LABELS}
    for lab, lst in hits.items():
        votes[lab] = min(len(lst), 3)  # cap
    mx = max(votes.values()) if votes else 0
    if mx == 0:
        return {lab: 0.0 for lab in LABELS}
    return {lab: votes[lab]/mx for lab in LABELS}
