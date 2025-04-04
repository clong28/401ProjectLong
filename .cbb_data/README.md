### College Basketball Data
---
Datasets for seasons 2013-2025
---
### Content
- Data from the 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 Division I college basketball seasons. ```cbb.csv``` has seasons 2013-2019 and seasons 2021-2024 combined. The 2020 season's data set is kept separate from the other seasons, because there was no postseason due to the Coronavirus. ```cbb25``` contains regular season data up to 3/28/2025 including the final rankings before the start of the NCAA tournament.

---
### Variables
- **RK** (Only in cbb25 (until the tournament is complete)): The ranking of the team at the end of the regular season
- **TEAM:** The Division I college basketball school
- **CONF:** The Athletic Conference in which the school participates in (A10 = Atlantic 10, ACC = Atlantic Coast Conference, AE = America East, Amer = American, ASun = ASUN, B10 = Big Ten, B12 = Big 12, BE = Big East, BSky = Big Sky, BSth = Big South, BW = Big West, CAA = Colonial Athletic Association, CUSA = Conference USA, Horz = Horizon League, Ivy = Ivy League, MAAC = Metro Atlantic Athletic Conference, MAC = Mid-American Conference, MEAC = Mid-Eastern Athletic Conference, MVC = Missouri Valley Conference, MWC = Mountain West, NEC = Northeast Conference, OVC = Ohio Valley Conference, P12 = Pac-12, Pat = Patriot League, SB = Sun Belt, SC = Southern Conference, SEC = South Eastern Conference, Slnd = Southland Conference, Sum = Summit League, SWAC = Southwestern Athletic Conference, WAC = Western Athletic Conference, WCC = West Coast Conference)
- **G:** Number of games played
- **W:** Number of games won
- **ADJOE:** Adjusted Offensive Efficiency (An estimate of the offensive efficiency (points scored per 100 possessions) a team would have against the average Division I defense)
- **ADJDE:** Adjusted Defensive Efficiency (An estimate of the defensive efficiency (points allowed per 100 possessions) a team would have against the average Division I offense)
- **BARTHAG:** Power Rating (Chance of beating an average Division I team)
- **EFG_O:** Effective Field Goal Percentage Shot
- **EFG_D:** Effective Field Goal Percentage Allowed
- **TOR:** Turnover Percentage Allowed ((Percentage team's possessions that end in a turnover))
- **TORD:** Turnover Percentage Committed (Percentage of opponent's possessions that end in a turnover)
- **ORB:** Offensive Rebound Rate
- **DRB:** Offensive Rebound Rate Allowed (Opponent Offensive Rebound Rate)
- **FTR:** Free Throw Rate (How often the given team shoots Free Throws)
- **FTRD:** Free Throw Rate Allowed (Team's ability to draw fouls and convert free throws)
- **2P_O:** Two-Point Shooting Percentage
- **2P_D:** Two-Point Shooting Percentage Allowed (Opponent 2 point shooting percentage)
- **3P_O:** Three-Point Shooting Percentage
- **3P_D:** Three-Point Shooting Percentage Allowed (Opponent 3 point shooting percentage)
- **ADJ_T:** Adjusted Tempo (An estimate of the tempo (possessions per 40 minutes) a team would have against the team that wants to play at an average Division I tempo)
- **WAB:** Wins Above Bubble (The bubble refers to the cut off between making the NCAA March Madness Tournament and not making it)
- **POSTSEASON:** Round where the given team was eliminated or where their season ended (R68 = First Four, R64 = Round of 64, R32 = Round of 32, S16 = Sweet Sixteen, E8 = Elite Eight, F4 = Final Four, 2ND = Runner-up, Champion = Winner of the NCAA March Madness Tournament for that given year)
- **SEED:** Seed in the NCAA March Madness Tournament
- **YEAR:** Season

---
Acknowledgements
This data was downloaded from [https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset]. 
