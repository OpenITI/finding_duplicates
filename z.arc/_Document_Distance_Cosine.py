###########################################################
# The script compares texts from RELEASE with any other
# files form a specified folder and generates TSV data
# with results: 
# - known text (URI/filename)
# - unknown text (filename)
# - distance (TFIDF-based)
# - local path to known text
# - local path to unknown text
###########################################################

###########################################################
# VARIABLES ###############################################
###########################################################

metadataFile = "/Users/romanov/_OpenITI_TEMP/RELEASE/metadata/OpenITI_metadata_2021-2-5.csv"
releasePath = "/Users/romanov/_OpenITI_TEMP/RELEASE/"

folderToCompare = "/Users/romanov/_OpenITI_TEMP/RAWrabica005000-master/" 

foldersToCompare = [
    "/Users/romanov/_OpenITI_TEMP/RAWrabica005000-master/",
    "/Users/romanov/_OpenITI_TEMP/RAWrabica005000-master/"
]

maxChunkLen = 30000 # use only this number of tokens from a file (taken from the beginning)
maxLength = 2 # used to shorted the text before processing (speeds up analysis sugnificantly)
loadDataTestVar = 100 # 200000 # reduce this value for testing purposes; 200 --- will take 100 from known, and 100 from unknown --- and will run fast
minTFIDF = 0.022 # calculations are made on the vector of words with values higher than this parameter
minCosine = 0.75 # items with distances below this will not be included in the final data;
chunkLengths = 20000 # for chunking large matrices; no need to change this parameter


###########################################################

print("="*80)
print("Analysis of files from: " + folderToCompare)
print("="*80)

###########################################################
# VARIABLES ###############################################
###########################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import itertools

import os
import re

### necessary libraries
import csv
from datetime import datetime 

### OpenITI functions
from openiti.helper.ara import deNoise, normalize_ara_light

###########################################################
###  functions ############################################
###########################################################



startTime = datetime.now()

# loading all the data
def loaData(folderToCompare, limitForTest=20000):
    print(">> Loading OpenITI RELEASE metadata:")
    metadataDic = {}
    limCounter = limitForTest//2
    with open(metadataFile, 'r') as data:     
        for record in csv.DictReader(data, delimiter='\t'):
            if record["status"] == "pri":
                metadataDic[record["version_uri"]] = record
                metadataDic[record["version_uri"]]["path"] = record["local_path"].replace("../", releasePath)
                #input(record)
                limCounter -= 1
                if limCounter == 0:
                    break

    print("\tLoaded texts with URIs: ", len(metadataDic))
    ###
    lof = os.listdir(folderToCompare)
    print(">> Loading UnknownFolder metadata: %d" % len(lof))
    for f in lof[:(limitForTest//2)]:
        if not f.startswith("."):
            metadataDic[f] = {"path": folderToCompare + f}
    #print("\tLoaded texts without URIs (+/- 1 or 2): ", len(lof))
    print("Total number of texts: %d" % len(metadataDic))

    ### 
    print(">> Converting data into a Corpus:")
    print("\tEach text is reduced to %s tokens; the raw text is reduced to %d items before preprocessing" % (maxChunkLen, maxChunkLen*maxLength))
    docList = []
    docIdList = []

    counter = 0
    timeLoopStart = datetime.now()

    for docId, docData in metadataDic.items():
        counter += 1
        if counter % 1000 == 0:
            timeLoopPassed = datetime.now() - timeLoopStart
            print("\t\t%d" % counter, '\tTime passed (hh:mm:ss.ms) {}'.format(timeLoopPassed))
            timeLoopStart = datetime.now()

        with open(docData["path"], "r", encoding="utf8") as f1:
            doc = f1.read()

            # cut the text - the following allows to cut the time significantly by reducing the length of text before processing
            doc = doc.split(" ")[:maxChunkLen*maxLength]
            doc = " ".join(doc)

            # clean doc
            doc = deNoise(doc) # remove short vowels...
            doc = normalize_ara_light(doc) # normalize arabic text...
            doc = re.sub("<[^>]+>", " ", doc)
            doc = re.sub('[a-zA-Z]', ' ', doc)
            doc = re.sub('\W+', ' ', doc)
            doc = re.sub('\d+', ' ', doc)
            doc = re.sub('_+', ' ', doc)
            doc = re.sub(' +', ' ', doc)
            docSection = " ".join(doc.split(" ")[:maxChunkLen])

            docList.append(docSection)
            docIdList.append(docId)

    return(metadataDic, docList, docIdList)

def docDistance(metadataDic, docList, docIdList, folderToCompare, matrixType, distanceType): # docDistance(f, 2, 0.85, (1, 1))
    print("="*80)
    print("Analysis of files from: " + folderToCompare)
    print("="*80)

    print("="*80)
    print(">> generating TFIDF data and Cosine Distances for: %d texts" % len(metadataDic))
    print("="*80)

    # MAIN PART: calculate tfidf for all loaded publications and distances
    print("\tgenerating tfidf matrix...")
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2, max_df=0.85, stop_words=[])
    countVectorized = vectorizer.fit_transform(docList)
    if matrixType == "TFIDF":
        tfidfTransformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        vectorized = tfidfTransformer.fit_transform(countVectorized)  # generates a sparse matrix
    elif matrixType == "DD":
        vectorized = countVectorized
    else:
        input("matrixType incorrect: must be `TFIDF` or `DD`")

    # PART 2: processing distances --- for both publications and page clusters
    # - something similar to this approach must be implemented above.
    print("\tgenerating cosine distances matrix...")
    print("\t\tconverting sparse matrix to compressed sparse row matrix")
    vectorizedCSR = vectorized.tocsr()
    print("\t\tpreparing grouping data...")
    nTexts = list(vectorizedCSR.get_shape())[0]
    indexList = list(range(nTexts))
    print("\t\tnumber of items: ", len(indexList))
    print("\t\tslicing of the matrix if above: ", chunkLengths)

    # chunking the matrix, if it is too large
    if len(indexList) > chunkLengths:
        print("\t\tthe chunk is shorter")
        n = chunkLengths
        indexGroupsList = [indexList[i * n:(i + 1) * n] for i in range((len(indexList) + n - 1) // n)]
        indexGroupsList = list(itertools.combinations(indexGroupsList, 2))
        print("\t\tgroups number: ", len(indexGroupsList))
        # input(indexGroupsList)
    else:
        print("\t\tthe chunk is longer")
        indexA = indexList[:len(indexList)//2]
        indexB = indexList[len(indexList)//2:]
        indexGroupsList = [(indexA, indexB)]
        print("\t\tgroups number: ", len(indexGroupsList))
        # input(indexGroupsList)

    print("\tnumber of groups: %d" % len(indexGroupsList))
    print("\tprocessing cosine distances data...")
    counter = len(indexGroupsList)

    # processing the matrix (or matrices, if the initial was too large)
    # running on DD matrix
    for i in indexGroupsList:
        print("\t\tgroup: %d" % counter)
        counter -= 1

        distances = {}

        i = list(i)
        ar = i[0] + i[1]
        indexingArray = np.array(ar)
        vectorizedCSRtemp = vectorizedCSR[indexingArray, :]
        docIdListTemp = docIdList[i[0][0]:i[0][-1]+1] + docIdList[i[1][0]:i[1][-1]+1]

        if distanceType == "cosine":
            distanceMatrixTemp = cosine_similarity(vectorizedCSRtemp)  # creates a dense matrix
        elif distanceType == "euclidean":
            distanceMatrixTemp = euclidean_distances(vectorizedCSRtemp) # creates a dense matrix
        else:
            input("distanceType incorrect: must be `cosine` or `euclidean`...")

        distanceTable = pd.DataFrame(distanceMatrixTemp)
        #print("\t\tcosineTable Shape: ", cosineTable.shape)
        distanceTable.columns = docIdListTemp
        distanceTable.index = docIdListTemp
        finalTable = distanceTable

        for column in finalTable:
            tempDF = finalTable[[column]].copy()
            tempDF = tempDF.loc[tempDF[column] >= minCosine]
            tempDF = tempDF.sort_values(by=[column], ascending=False)
            tempDF[column] = tempDF[column].round(decimals=5)
            #input(tempDF[column])
            tempDic = tempDF.to_dict()

            if column in distances:
                distances[column].update(tempDic)
            else:
                distances[column] = tempDic

        # input(distances)
        #print("\tsaving cosine distances data...")
        print("\tAggregating results into TSV format...")
        tsvData = ["known\tunknown\tDD_Cos\tstatus"]

        finalCountDown = len(metadataDic)
        print("\tTo process: %d" % finalCountDown)
        for column, results in distances.items():
            finalCountDown -= 1
            if finalCountDown % 1000 == 0:
                print("\t\t%d remaining..." % finalCountDown)
            # check if the text has a URI
            #if re.search("^\d\d\d\d\w+\.\w+", column): # 
            for text1, results1 in results.items():
                for text2, distance in results1.items():
                    if re.search("^\d\d\d\d\w+\.\w+", text2):
                        pass
                    else:
                        #print("\t", text2, "\t", distance)
                        val = [text1, text2, str(distance), "review/true/false"]
                        tsvData.append("\t".join(val))

        finalData = "\n".join(tsvData)
        suffix = folderToCompare.split("/")[-2]
        fileFinal = "results_%s_%s_%s_%dchunks.tsv" % (matrixType, distanceType, suffix, maxChunkLen)
        print("\tSaving results into a TSV file: " + fileFinal)
        with open(fileFinal, "w", encoding="utf8") as f9:
            f9.write(finalData)

###########################################################
# RUNNING ALL #############################################
###########################################################

for folder in foldersToCompare:
    metadataDic, docList, docIdList  = loaData(folderToCompare, loadDataTestVar)
    docDistance(metadataDic, docList, docIdList, folder, "TFIDF", "cosine")
    docDistance(metadataDic, docList, docIdList, folder, "DD", "cosine")
    docDistance(metadataDic, docList, docIdList, folder, "DD", "euclidean")

timeElapsed = datetime.now() - startTime
print('Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))
