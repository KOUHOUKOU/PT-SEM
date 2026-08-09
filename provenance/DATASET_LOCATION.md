# Kaggle Dataset Location and Citation

Creator: Vladislav Shufinskiy (`brains14482`)

Dataset title: *NBA WNBA play-by-play and shots data*

Dataset identifier: `brains14482/nba-playbyplay-and-shotdetails-data-19962021`

Version: 8 (released 2025-06-26; downloaded 2026-05-12)

Version-specific URL:

```text
https://www.kaggle.com/datasets/brains14482/nba-playbyplay-and-shotdetails-data-19962021/versions/8
```

License: Apache 2.0

The paper-ready reference and BibTeX entry are in
`KAGGLE_DATA_SOURCE_CITATION.md` and `kaggle_dataset_citation.bib`.

Downloaded with:

```powershell
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('brains14482/nba-playbyplay-and-shotdetails-data-19962021'))"
```

Local cache path:

```text
C:\Users\ROG\.cache\kagglehub\datasets\brains14482\nba-playbyplay-and-shotdetails-data-19962021\versions\8
```

For the FOUL/FTA/FTM quarter-level replication, start with these NBA.com
play-by-play files:

```text
nbastats_2015.csv
nbastats_2016.csv
nbastats_2017.csv
nbastats_2018.csv
nbastats_2019.csv
nbastats_2020.csv
```

They contain the fields needed for aggregation, including `GAME_ID`,
`EVENTMSGTYPE`, `EVENTMSGACTIONTYPE`, `PERIOD`, event descriptions, and team
IDs. These six files correspond to seasons 2015-16 through 2020-21: for
example, `nbastats_2016.csv` starts on 2016-10-25 and has game IDs like
`21600001`, so it is the 2016-17 season.
