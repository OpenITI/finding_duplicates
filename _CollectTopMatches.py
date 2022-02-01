import re, os
import pandas as pd

######################################################################
# PARAMETERS #########################################################
######################################################################

cosMin = 0.93

######################################################################

files = ["results_RAW_ABLibrary.tsv", "results_RAW_Masaha.tsv", "results_raw_ShamelaY19.tsv",
    "results_RAWrabica005000.tsv", "results_RAWrabica010000.tsv", "results_RAWrabica015000.tsv",
    "results_RAWrabica020000.tsv", "results_RAWrabica025000.tsv", "results_RAWrabica030000.tsv",
    "results_RAWrabica035000.tsv", "results_RAWrabica040000.tsv", "results_RAWrabica045000.tsv",
    "results_RAWrabicaRafed.tsv", "results_RAWrabicaSham19Y.tsv", "results_RAWrabicaShamAYtxt1.tsv",
    "results_RAWrabicaShamAYtxt2.tsv", "results_RAWrabicaShamAYtxt3.tsv"]

final = ["known\tunknown\tDDC\tDDE\tTFC\tLib"]

for f in files:
    with open(f, "r", encoding="utf8") as f1:
        data = f1.read().split("\n")
        for d in data[1:]:
            if "~" in d:
                pass
            else:
                final.append(d + "\t" + f)

with open("_filtered_matches.tsv", "w", encoding="utf8") as f9:
    f9.write("\n".join(final))