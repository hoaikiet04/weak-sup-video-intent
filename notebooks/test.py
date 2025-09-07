from transformers import pipeline

clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["KIS", "How-to", "Music", "News", "Sports", "Review"]

q = "cách sửa lỗi wifi Windows 11"
res = clf(q, candidate_labels=labels, multi_label=False)
print(res["labels"][0], res["scores"][0])