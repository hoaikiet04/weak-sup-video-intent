# snorkel_setup.py
ABSTAIN = -1
LABELS = ["KIS","How-to","Music","News","Sports","Review","Entertainment","Other"]
L2I = {lab:i for i,lab in enumerate(LABELS)}
I2L = {i:lab for lab,i in L2I.items()}