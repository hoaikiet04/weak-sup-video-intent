# lfs_text.py
import re
from snorkel.labeling import labeling_function
from snorkel_setup import ABSTAIN, L2I

# Regex helpers
RX = {
    "Music": re.compile(r"\blyrics?\b|karaoke|\bmv\b|official\s+mv|official\s+audio|\blofi(\s+chill)?\b|\bnhạc\b|\b(remix|instrumental|playlist|ost|soundtrack)\b", re.I),
    "How-to": re.compile(r"\bhow\s*to\b|tutorial|hướng\s*dẫn|\bcách\b|\bfix\b|\bsửa\b|install|setup|cài\s*đặt|khắc\s*phục|guide|step\s*by\s*step", re.I),
    "Sports": re.compile(r"\b(highlight|highlights)\b|\bfull\s*match\b|\bvs\b|\btrận\b|world\s*cup|premier\s*league|champions\s*league|\blive\b|\bgoals?\b|\bscore\b", re.I),
    "News": re.compile(r"\bnews\b|breaking|thời\s*sự|bản\s*tin|tin\s*tức|evening\s*news|morning\s*news", re.I),
    "Review": re.compile(r"\breview\b|đánh\s*giá|so\s*sánh|\bunboxing\b|\bgiá\b|\bhands?-?on\b|\btrên\s*tay\b|\bcompare\b", re.I),
    "KIS": re.compile(r"trailer|official\s*trailer|episode|\bep\.?\b|season|\btập\s*\d+\b|vietsub|full\s*movie|\bchapter\b|\bchương\b|\bphần\s*\d+\b", re.I),
    "Entertainment": re.compile(r"\bvlog\b|reaction|pranks?|challenge|funny|memes?|skit|talk\s*show|podcast|live\s*stream|livestream|streaming|asmr|parody", re.I),
}

# Ambiguity helpers (tie-breaky)
RX_SO_SANH_VS   = re.compile(r"so\s*sánh.*\bvs\b", re.I)
RX_EP_INDICATORS= re.compile(r"\btập\s*\d+|\bepisode\b|\bep\.?\b|\bseason\b|\bphần\s*\d+|\bchapter\b|\bchương\b|\bvietsub\b", re.I)
RX_MUSIC_STRONG = re.compile(r"\blyrics?\b|\bkaraoke\b|\bofficial\s+mv\b|\bmv\b|\bofficial\s+audio\b", re.I)
RX_SPORTS_STRONG= re.compile(r"\b(highlight|highlights|full\s*match|goal|goals|world\s*cup|premier\s*league|champions\s*league|live\b)", re.I)

def _label_if(rx, txt, lab):
    return L2I[lab] if rx.search(txt) else ABSTAIN

@labeling_function()
def lf_music(x):
    return _label_if(RX["Music"], x.text, "Music")

@labeling_function()
def lf_howto(x):
    return _label_if(RX["How-to"], x.text, "How-to")

@labeling_function()
def lf_sports(x):
    # tránh dính 'vs' ở review so sánh
    if RX_SO_SANH_VS.search(x.text):
        return ABSTAIN
    return _label_if(RX["Sports"], x.text, "Sports")

@labeling_function()
def lf_news(x):
    return _label_if(RX["News"], x.text, "News")

@labeling_function()
def lf_review(x):
    # nếu có "so sánh ... vs ..." ưu tiên review
    if RX_SO_SANH_VS.search(x.text):
        return L2I["Review"]
    return _label_if(RX["Review"], x.text, "Review")

@labeling_function()
def lf_kis(x):
    # tie với music: nếu có tập/season và KHÔNG có music-strong → KIS
    if RX_EP_INDICATORS.search(x.text) and not RX_MUSIC_STRONG.search(x.text):
        return L2I["KIS"]
    return _label_if(RX["KIS"], x.text, "KIS")

@labeling_function()
def lf_entertainment(x):
    return _label_if(RX["Entertainment"], x.text, "Entertainment")

# (tuỳ chọn) LF 'Other' chỉ dùng khi muốn ép phong bì rác/quảng cáo
@labeling_function()
def lf_other_ads(x):
    if re.search(r"\bofficial\s*ad\b|\b(ad|promo|sale|discount)\b|giảm\s*giá|click\s*here", x.text, re.I):
        return L2I["Other"]
    return ABSTAIN

LFS = [lf_music, lf_howto, lf_sports, lf_news, lf_review, lf_kis, lf_entertainment, lf_other_ads]
