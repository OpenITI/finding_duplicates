import os

folder = "/Users/romanov/_OpenITI_TEMP/RAWrabicaSham19Y/"
lof = os.listdir(folder)

for f in lof:
    if not f.startswith("."):
        print(folder + f)
        with open(folder + f, "r", encoding="utf8") as f1:
            test = f1.read()