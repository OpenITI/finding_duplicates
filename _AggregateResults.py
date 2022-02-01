import re, os
import pandas as pd

######################################################################
# PARAMETERS #########################################################
######################################################################

cosMin = 0.93

######################################################################

sortingHat = {
    "DD_cosine": 0,
    "DD_euclidean": 1,
    "TFIDF_cosine": 2
}

folder = "./results/"
lof = sorted(os.listdir(folder))

libraries = ["RAW_ABLibrary", "RAW_Masaha", "raw_ShamelaY19", "RAWrabica005000", "RAWrabica010000", "RAWrabica015000", "RAWrabica020000", "RAWrabica025000", "RAWrabica030000", "RAWrabica035000", "RAWrabica040000", "RAWrabica045000", "RAWrabicaRafed", "RAWrabicaSham19Y", "RAWrabicaShamAYtxt1", "RAWrabicaShamAYtxt2", "RAWrabicaShamAYtxt3"]

for libVar in libraries:
    results = {} # key: known>unknown; val: [DDcos, TFIDFcos, DDeuc]
    print(libVar)
    for f in lof:
        if f.startswith("results") and libVar in f:
            print("\t", f)
            vars = re.sub("_RAW_|_raw_", "_Raw", f)
            vars = vars.split("_")
            dist = vars[3]
            matr = vars[2]
            resultType = vars[2] + "_" + vars[3]
            
            with open(folder + f, "r", encoding="utf8") as infile:
                next(infile)

                for l in infile:
                    l = l.split("\t")
                    val = float(l[2])

                    if dist == "cosine" and val >= cosMin:
                        key = l[0] + "\t" + l[1]
                        if key in results:
                            results[key][sortingHat[resultType]] = l[2]
                        else:
                            #results[key] = ["DDC", "DDE", "TFC", libVar]
                            results[key] = ["~", "~", "~"]
                            results[key][sortingHat[resultType]] = l[2]
                            #input(results)

                    elif dist == "euclidean":
                        key = l[0] + "\t" + l[1]
                        if key in results:
                            results[key][sortingHat[resultType]] = l[2]
                        else:
                            results[key] = ["~", "~", "~"]
                            results[key][sortingHat[resultType]] = l[2]
                            #input(results)

    final = ["known\tunknown\tDDC\tDDE\tTFC"]
    for k,v in results.items():
        final.append(k + "\t" + "\t".join(v))
        #print(final)
        #input(v)

    with open("results_%s.tsv" % libVar, "w", encoding="utf8") as f9:
        f9.write("\n".join(final))

