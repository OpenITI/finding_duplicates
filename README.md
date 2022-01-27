# finding_duplicates

The script compares texts from OpenITI RELEASE with any other files form a specified folder and generates TSV data with results: 

- known text (URI/filename)
- unknown text (filename)
- distance (TFIDF-based)
- local path to known text
- local path to unknown text

Full match is usually above 0.9

The script requires:

- the OpenITI RELEASE data (path to folder)
- the OpenITI RELEASE metadata file (path to file)
- data to analyze (path to folder; assumed that there are only text files; the format of those files doues not matter)

