from sklearn.metrics import confusion_matrix, cohen_kappa_score

M = [0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0]
H = [0,1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0]

fp_indices = [i for i, (m, h) in enumerate(zip(M, H)) if m == 1 and h == 1]
print(f"FP indices (0-based): {fp_indices}")
print()

with open("wrime-ver2.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in fp_indices:
    sentence = lines[i + 1].split("\t")[0]
    print(f"{i + 1}: {sentence}")