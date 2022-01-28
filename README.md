# finding_duplicates

The script compares texts from OpenITI RELEASE with any other files form a specified folder and generates TSV data with results: 

- known text (URI/filename)
- unknown text (filename)
- distance (TFIDF-based)
- local path to known text
- local path to unknown text

Full match is usually above 0.9 (often about 0.99, but not always)

The script requires:

- the OpenITI RELEASE data (path to folder)
- the OpenITI RELEASE metadata file (path to file)
- data to analyze (path to folder; assumed that there are only text files; the format of those files doues not matter)


Script report example:
```

XXXXXXX@XXXXXXX finding_duplicates % python3 TFIDF_Distance.py
================================================================================
Analysis of files from: /Users/XXXXXXX/_OpenITI_TEMP/RAWrabica005000-master/
================================================================================
>> Loading OpenITI RELEASE metadata:
	Loaded texts with URIs:  6338
>> Loading UnknownFolder metadata: 5002
Total number of texts: 11339
>> Converting data into a Corpus:
	Each text is reduced to 30000 tokens; the raw text is reduced to 60000 items before preprocessing
		1000 	Time passed (hh:mm:ss.ms) 0:00:17.383660
		2000 	Time passed (hh:mm:ss.ms) 0:00:21.795654
		3000 	Time passed (hh:mm:ss.ms) 0:00:27.775684
		4000 	Time passed (hh:mm:ss.ms) 0:00:30.795721
		5000 	Time passed (hh:mm:ss.ms) 0:00:34.260374
		6000 	Time passed (hh:mm:ss.ms) 0:00:39.846982
		7000 	Time passed (hh:mm:ss.ms) 0:00:36.973594
		8000 	Time passed (hh:mm:ss.ms) 0:00:32.474960
		9000 	Time passed (hh:mm:ss.ms) 0:00:34.537479
		10000 	Time passed (hh:mm:ss.ms) 0:00:34.285643
		11000 	Time passed (hh:mm:ss.ms) 0:00:30.608689
================================================================================
>> generating TFIDF data and Cosine Distances for: 11339 texts
================================================================================
	generating tfidf matrix...
	generating cosine distances matrix...
		converting sparse matrix to compressed sparse row matrix
		preparing grouping data...
		number of items:  11339
		slicing of the matrix if above:  20000
		the chunk is longer
		groups number:  1
	number of groups: 1
	processing cosine distances data...
		group: 1
	Aggregating results into TSV format...
	To process: 11339
		11000 remaining...
		10000 remaining...
		9000 remaining...
		8000 remaining...
		7000 remaining...
		6000 remaining...
		5000 remaining...
		4000 remaining...
		3000 remaining...
		2000 remaining...
		1000 remaining...
		0 remaining...
	Saving results into a TSV file...
Time elapsed (hh:mm:ss.ms) 0:10:31.442058

```

## OpenITI RAW repos:


- https://github.com/OpenITI/RAWrabica005000.git
- https://github.com/OpenITI/RAWrabica010000.git
- https://github.com/OpenITI/RAWrabica015000.git
- https://github.com/OpenITI/RAWrabica015000.git
- https://github.com/OpenITI/RAWrabica020000.git
- https://github.com/OpenITI/RAWrabica025000.git
- https://github.com/OpenITI/RAWrabica030000.git
- https://github.com/OpenITI/RAWrabica035000.git
- https://github.com/OpenITI/RAWrabica040000.git
- https://github.com/OpenITI/RAWrabica045000.git

- https://github.com/OpenITI/raw_SHAM19Y.git
- https://github.com/OpenITI/RAWrabicaSham19Y.git
- https://github.com/OpenITI/raw_ShamelaY19.git
- https://github.com/OpenITI/RAWrabicaGRAR.git
- https://github.com/OpenITI/RAWrabica_ManchesterFiles.git
- https://github.com/OpenITI/RAWrabicaRafed.git
- https://github.com/OpenITI/RAWrabicaShamAY1.git
- https://github.com/OpenITI/RAWrabicaShamAY2.git
- https://github.com/OpenITI/RAWrabicaShamAY3.git









