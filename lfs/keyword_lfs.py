import re
from typing import Optional, Dict, List

LABELS = ["KIS", "How-to", "Music", "News", "Sports", "Review"]

# Simple keyword-based labeling functions (LFs). Return a single-label vote or None.
def lf_music(q: str) -> Optional[str]:
    return "Music" if re.search(r"\blyrics?\b|karaoke|\bmv\b|official audio|\bbeat\b|\bnhạc\b|\blyric\b", q, re.I) else None

def lf_howto(q: str) -> Optional[str]:
    return "How-to" if re.search(r"how to|tutorial|hướng dẫn|\bcách\b|\bfix\b|\bsửa\b|mẹo", q, re.I) else None

def lf_sports(q: str) -> Optional[str]:
    return "Sports" if re.search(r"\bvs\b|highlight|full match|\btrận\b|bán kết|chung kết|world cup|premier league|\blive\b", q, re.I) else None

def lf_news(q: str) -> Optional[str]:
    return "News" if re.search(r"\bnews\b|breaking|thời sự|bản tin|\bnóng\b|evening news|morning news", q, re.I) else None

def lf_review(q: str) -> Optional[str]:
    return "Review" if re.search(r"\breview\b|đánh giá|so sánh|\bunboxing\b|\bgiá\b", q, re.I) else None

def lf_kis(q: str) -> Optional[str]:
    return "KIS" if re.search(r"trailer|episode|\bep\.?\b|season|tập\s?\d+|vietsub|official mv|channel", q, re.I) else None

LFS = [lf_music, lf_howto, lf_sports, lf_news, lf_review, lf_kis]

def votes_dict(q: str) -> Dict[str, int]:
    votes = {lab: 0 for lab in LABELS}
    for f in LFS:
        lab = f(q)
        if lab:
            votes[lab] += 1
    return votes

def predict_rules_only(q: str) -> str:
    votes = votes_dict(q)
    best = max(votes, key=votes.get)
    return best if votes[best] > 0 else "Other"
