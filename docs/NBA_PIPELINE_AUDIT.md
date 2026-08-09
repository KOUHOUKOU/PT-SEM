# NBA pipeline audit

## Dataset and observation unit

The empirical pipeline uses version 8 of Vladislav Shufinskiy's Kaggle dataset *NBA WNBA play-by-play and shots data* (`brains14482/nba-playbyplay-and-shotdetails-data-19962021`), released 2025-06-26 under Apache-2.0. The ten raw files `nbastats_2015.csv` through `nbastats_2024.csv` correspond to 2015-16 through 2024-25.

The analysis unit is a team-quarter. Periods greater than four are excluded. The committed processed data contain 95,808 observations. The five variables shown in the paper are:

- `FOUL`: non-offensive fouls drawn, assigned to `PLAYER2_TEAM_ID`, excluding action codes 4 and 26;
- `FTA`: free-throw attempts;
- `FTM`: made free throws;
- `PERS_FOUL_DRAWN` (`PERS` in the figure): personal fouls drawn, action codes 1, 27, and 28;
- `LOOSE_BALL_FOUL_DRAWN` (`LOOSE` in the figure): loose-ball fouls drawn, action code 3.

The reference DAG is `FOUL -> FTA`, `FTA -> FTM`, `FOUL -> PERS`, and `FOUL -> LOOSE`.

## Recovered program chain

1. `build_quarter_counts.py` parses play-by-play records and builds core quarter counts.
2. `build_nba_five_var_extension.py` produces the five-variable ten-season team-quarter files.
3. `run_expanded_candidate_optimization.py` fits the six-family DP-BIC model.
4. `run_nba_five_var_baseline_comparison_fixed.py` and companion baseline programs run ODS, PC-RCIT, PB-SCM, and PB-SCM-PGF.
5. `summarize_nba_extension.py` writes recovery and parameter tables.
6. `make_nba_final_figures.py` produces Figures 5 and 6.

All ten fitted graph JSON files, processed CSVs, coefficient tables, working-family tables, recovery tables, exact paper figures, and the complete original `Basketball/src` Python source set were recovered. Active copies use repository-relative defaults or explicit CLI arguments instead of the original workstation paths. The main method exactly recovers the directed DAG in 9/10 seasons and its skeleton in 10/10. In 2022-23 only the `FOUL`--`LOOSE` direction is reversed.

The older files in `provenance/` mention a six-season intermediate analysis. They are retained as original historical records; this audit and the ten committed data files supersede their scope statement for the current manuscript.
