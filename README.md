# 🏏 Vijay Hazare Trophy 2025-26 Dashboard

A comprehensive interactive dashboard analyzing the complete Vijay Hazare Trophy 2025-26 tournament built with Python, Dash, and Plotly.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.14-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🏆 Tournament Summary

| Item              | Details                                              |
| ----------------- | ---------------------------------------------------- |
| **Champion**      | **VIDARBHA**                                         |
| **Runner-up**     | Saurashtra                                           |
| **Final Score**   | VID 317-8 (50 ov) beat SAUR 279 (48.5 ov) by 38 runs |
| **Total Matches** | 135 (112 Group + 16 Plate + 7 Knockout)              |
| **Teams**         | 38 (32 Main + 6 Plate)                               |
| **Total Runs**    | 66,026                                               |
| **Total Wickets** | 1,897                                                |

---

## 📊 Dashboard Pages

| #   | Page                       | Description                                     |
| --- | -------------------------- | ----------------------------------------------- |
| 1   | 🏠 **Tournament Overview** | KPIs, standings, champion card, top teams       |
| 2   | 📊 **Team Comparison**     | Quadrant analysis, radar charts, rankings       |
| 3   | 🏏 **Batting Analysis**    | Run scoring, boundaries, top 3 dependency       |
| 4   | 🎯 **Bowling Analysis**    | Economy, wickets, death bowling                 |
| 5   | 🎲 **Toss & Venue Impact** | Toss decisions, venue patterns, chase vs defend |
| 6   | ⚔️ **Match Situations**    | Win margins, pressure wins, closest finishes    |
| 7   | 🧤 **Dismissal Patterns**  | Caught, bowled, LBW analysis                    |
| 8   | 🎖️ **Qualified Teams**     | Deep dive into 10 qualified teams               |
| 9   | 🏆 **Champion's Journey**  | Vidarbha's complete path to victory             |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Download/Clone the Project

Create a folder and place all files in this structure:

```
vht-dashboard/
│
├── data/
│   ├── matches.csv
│   ├── team_stats.csv
│   ├── player_batting.csv
│   ├── player_bowling.csv
│   ├── venue_stats.csv
│   └── standings.csv
│
├── assets/
│   └── style.css
│
├── app.py
├── data_processor.py
├── requirements.txt
└── README.md
```

### Step 2: Install Dependencies

Open terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install dash dash-bootstrap-components pandas numpy plotly
```

### Step 3: Run the Dashboard

```bash
python app.py
```

### Step 4: View the Dashboard

Open your web browser and go to:

```
http://127.0.0.1:8050
```

That's it! 🎉

---

## 📁 Project Structure

```
vht-dashboard/
│
├── data/                          # Data files (CSV)
│   ├── matches.csv                # 135 match records
│   ├── team_stats.csv             # 38 teams, 69 metrics each
│   ├── player_batting.csv         # 2,443 batting records
│   ├── player_bowling.csv         # 1,669 bowling records
│   ├── venue_stats.csv            # 23 venue statistics
│   └── standings.csv              # Group standings
│
├── assets/                        # Static files
│   └── style.css                  # Custom styling (653 lines)
│
├── app.py                         # Main dashboard (2,521 lines)
├── data_processor.py              # ETL script (reference)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📈 Data Summary

### matches.csv (135 records)

| Column                     | Description                                |
| -------------------------- | ------------------------------------------ |
| match_id                   | Unique match identifier                    |
| team1, team2               | Playing teams                              |
| team1_runs, team2_runs     | Scores                                     |
| winner                     | Match winner                               |
| win_type                   | Won by runs/wickets                        |
| win_margin                 | Margin of victory                          |
| venue                      | Match venue                                |
| stage                      | Group Stage/Quarter Final/Semi Final/Final |
| toss_winner, toss_decision | Toss details                               |

### team_stats.csv (38 teams, 69 columns)

Key metrics include:

- Match stats: matches_played, wins, losses, win_pct, nrr
- Batting: total_runs, avg_score, team_sr, boundary_pct, fifties, hundreds
- Bowling: economy, bowling_avg, bowling_sr, total_wickets
- Advanced: top3_dependency, chase_success_rate, pressure_wins, balanced_index

### player_batting.csv (2,443 records)

| Column         | Description      |
| -------------- | ---------------- |
| Batter         | Player name      |
| Team           | Team code        |
| Runs, Balls    | Batting figures  |
| 4s, 6s         | Boundaries       |
| SR             | Strike rate      |
| dismissal_type | How they got out |

### player_bowling.csv (1,669 records)

| Column         | Description     |
| -------------- | --------------- |
| Bowler         | Player name     |
| Team           | Team code       |
| Overs, Maidens | Bowling figures |
| Runs, Wickets  | Conceded/taken  |
| Economy        | Economy rate    |

---

## 🎨 Color Scheme

| Color                 | Hex Code  | Usage               |
| --------------------- | --------- | ------------------- |
| Primary (Red)         | `#e63946` | Highlights, alerts  |
| Secondary (Navy)      | `#1d3557` | Headers, sidebar    |
| Tertiary (Steel Blue) | `#457b9d` | Secondary elements  |
| Success (Teal)        | `#2a9d8f` | Positive metrics    |
| Warning (Orange)      | `#f4a261` | Warnings, attention |
| Gold                  | `#ffb703` | Champion, qualified |
| Purple                | `#6a4c93` | Accents             |
| Background            | `#f8f9fa` | Page background     |

---

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution:** Make sure all packages are installed:

```bash
pip install dash dash-bootstrap-components pandas numpy plotly
```

### Issue: "File not found" error for CSV files

**Solution:** Make sure the `data/` folder exists and contains all 6 CSV files.

### Issue: Dashboard not loading

**Solution:**

1. Check if port 8050 is available
2. Try a different port:

```python
app.run_server(debug=True, port=8051)
```

### Issue: Charts not displaying

**Solution:** Clear browser cache or try a different browser (Chrome recommended).

---

## 📊 Key Insights from the Data

1. **Vidarbha's Dominance**: 80% win rate, balanced batting (avg 284.9) and bowling (economy 5.73)

2. **Top Performers**:
   - Most Runs: Aman Mokhade (814 runs)
   - Most Wickets: Ankur Panwar (25 wickets)

3. **Toss Impact**: Teams choosing to bowl first won 52% of matches

4. **High Scoring**: 124 centuries scored across the tournament

5. **Pressure Performers**: Teams with better death bowling (economy < 7) had higher win rates

---

## 🛠️ Technologies Used

| Technology                    | Purpose                    |
| ----------------------------- | -------------------------- |
| **Python 3.8+**               | Core programming language  |
| **Dash 2.14**                 | Web application framework  |
| **Plotly 5.18**               | Interactive visualizations |
| **Pandas 2.1**                | Data manipulation          |
| **NumPy 1.26**                | Numerical operations       |
| **Dash Bootstrap Components** | UI styling                 |

---

## 📝 License

This project is for educational and portfolio purposes. Data sourced from publicly available cricket statistics.

---

## 👨‍💻 Author

**Sports Analytics Project**

Built as part of a comprehensive cricket analytics portfolio demonstrating:

- End-to-end data processing
- Interactive dashboard development
- Sports statistics analysis
- Data visualization best practices

---

## 🙏 Acknowledgments

- BCCI for organizing the Vijay Hazare Trophy
- All participating teams and players
- Cricket statistics providers

---

## 📞 Support

If you encounter any issues:

1. Check the Troubleshooting section above
2. Verify all files are in the correct folder structure
3. Ensure Python 3.8+ is installed

---

**🏆 Congratulations to VIDARBHA - Vijay Hazare Trophy 2025-26 Champions! 🏆**
