"""
Vijay Hazare Trophy 2024-25 Data Processor
==========================================
Parses all 135 matches (119 main + 16 plate) and creates consolidated CSV files.

Output Files:
- matches.csv: All match-level data
- team_stats.csv: Aggregated team statistics with all metrics
- player_batting.csv: All player batting records
- player_bowling.csv: All player bowling records

Run this script once to generate processed data for the dashboard.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - Update these if your data is in a different location
RAW_DATA_PATH = "VHT_FIN1/data"
PLATE_DATA_PATH = "VHT_FIN1/plate_data"
OUTPUT_PATH = "data"

# Team name mappings (short code to full name)
TEAM_NAMES = {
    'AP': 'Andhra Pradesh',
    'ASM': 'Assam',
    'BEN': 'Bengal',
    'BRD': 'Baroda',
    'CDG': 'Chandigarh',
    'CG': 'Chhattisgarh',
    'DEL': 'Delhi',
    'GOA': 'Goa',
    'GUJ': 'Gujarat',
    'HAR': 'Haryana',
    'HP': 'Himachal Pradesh',
    'HYD': 'Hyderabad',
    'JHKD': 'Jharkhand',
    'JK': 'Jammu & Kashmir',
    'KAR': 'Karnataka',
    'KER': 'Kerala',
    'MAH': 'Maharashtra',
    'MP': 'Madhya Pradesh',
    'MUM': 'Mumbai',
    'ODSA': 'Odisha',
    'PDC': 'Puducherry',
    'PUN': 'Punjab',
    'RAJ': 'Rajasthan',
    'RLYS': 'Railways',
    'SAUR': 'Saurashtra',
    'SER': 'Services',
    'SKM': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TRI': 'Tripura',
    'UP': 'Uttar Pradesh',
    'UTK': 'Uttarakhand',
    'VID': 'Vidarbha',
    # Plate teams
    'BIH': 'Bihar',
    'MNP': 'Manipur',
    'NGL': 'Nagaland',
    'MGLY': 'Meghalaya',
    'ARNP': 'Arunachal Pradesh',
    'MIZ': 'Mizoram'
}

# Group assignments
GROUPS = {
    'Group A': ['KAR', 'MP', 'KER', 'JHKD', 'TN', 'TRI', 'RAJ', 'PDC'],
    'Group B': ['UP', 'VID', 'BRD', 'BEN', 'JK', 'HYD', 'ASM', 'CDG'],
    'Group C': ['PUN', 'MUM', 'MAH', 'CG', 'HP', 'GOA', 'UTK', 'SKM'],
    'Group D': ['DEL', 'SAUR', 'HAR', 'RLYS', 'GUJ', 'ODSA', 'AP', 'SER'],
    'Plate': ['BIH', 'MNP', 'NGL', 'MGLY', 'ARNP', 'MIZ']
}

# Knockout dates for match stage identification
KNOCKOUT_DATES = {
    'Jan 12': 'Quarter Final',
    'Jan 13': 'Quarter Final',
    'Jan 15': 'Semi Final',
    'Jan 16': 'Semi Final',
    'Jan 18': 'Final'
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_team_group(team_code):
    """Get the group for a team"""
    for group, teams in GROUPS.items():
        if team_code in teams:
            return group
    return 'Unknown'


def parse_score(score_str):
    """
    Parse score string like 'AP 298-8  (50 Ov)' or 'DEL 300-6  (37.4 Ov)'
    Returns: (team, runs, wickets, overs)
    """
    try:
        # Clean the string
        score_str = score_str.strip()
        
        # Pattern: TEAM RUNS-WICKETS (OVERS Ov)
        pattern = r'([A-Z]+)\s+(\d+)-(\d+)\s+\((\d+\.?\d*)\s*Ov\)'
        match = re.match(pattern, score_str)
        
        if match:
            team = match.group(1)
            runs = int(match.group(2))
            wickets = int(match.group(3))
            overs = float(match.group(4))
            return team, runs, wickets, overs
        
        # Alternative pattern without wickets: TEAM RUNS (OVERS Ov)
        pattern2 = r'([A-Z]+)\s+(\d+)\s+\((\d+\.?\d*)\s*Ov\)'
        match2 = re.match(pattern2, score_str)
        
        if match2:
            team = match2.group(1)
            runs = int(match2.group(2))
            overs = float(match2.group(3))
            return team, runs, 10, overs  # Assume all out
            
        return None, None, None, None
    except:
        return None, None, None, None


def parse_toss_info(toss_str):
    """
    Parse toss string like 'Delhi won the toss and opt to Bowl'
    Returns: (toss_winner_name, toss_decision)
    """
    try:
        toss_str = toss_str.strip()
        
        # Pattern: TEAM won the toss and opt to BAT/BOWL
        pattern = r'(.+?)\s+won the toss and opt to\s+(Bat|Bowl)'
        match = re.search(pattern, toss_str, re.IGNORECASE)
        
        if match:
            toss_winner = match.group(1).strip()
            decision = match.group(2).capitalize()
            return toss_winner, decision
        
        return None, None
    except:
        return None, None


def parse_date(date_str):
    """
    Parse date string like 'Wed, Dec 24' or 'Mon, Jan 12'
    Returns: (day_name, month, day_num, stage)
    """
    try:
        date_str = date_str.strip().strip('"')
        parts = date_str.split(',')
        
        if len(parts) == 2:
            day_name = parts[0].strip()
            month_day = parts[1].strip()
            month_parts = month_day.split()
            
            if len(month_parts) == 2:
                month = month_parts[0]
                day_num = int(month_parts[1])
                
                # Determine stage based on date
                date_key = f"{month} {day_num}"
                stage = KNOCKOUT_DATES.get(date_key, 'Group Stage')
                
                return day_name, month, day_num, stage
        
        return None, None, None, 'Group Stage'
    except:
        return None, None, None, 'Group Stage'


def parse_dismissal(dismissal_str):
    """
    Parse dismissal string to extract dismissal type
    Types: caught, bowled, lbw, stumped, run out, not out, retired hurt, hit wicket, absent hurt
    """
    if pd.isna(dismissal_str):
        return 'unknown'
    
    dismissal_str = str(dismissal_str).strip().lower()
    
    if dismissal_str.startswith('not out') or dismissal_str == 'not out':
        return 'not out'
    elif dismissal_str.startswith('c ') or dismissal_str.startswith('c and b'):
        return 'caught'
    elif dismissal_str.startswith('b '):
        return 'bowled'
    elif dismissal_str.startswith('lbw'):
        return 'lbw'
    elif dismissal_str.startswith('st '):
        return 'stumped'
    elif dismissal_str.startswith('run out'):
        return 'run out'
    elif dismissal_str.startswith('retd') or 'retired' in dismissal_str:
        return 'retired hurt'
    elif dismissal_str.startswith('hit wicket'):
        return 'hit wicket'
    elif dismissal_str.startswith('abs') or 'absent' in dismissal_str:
        return 'absent hurt'
    elif dismissal_str.startswith('batting'):
        return 'batting'  # Did not bat
    else:
        return 'unknown'


def overs_to_balls(overs):
    """Convert overs (like 37.4) to total balls"""
    try:
        whole_overs = int(overs)
        partial = round((overs - whole_overs) * 10)
        return (whole_overs * 6) + partial
    except:
        return 0


def balls_to_overs(balls):
    """Convert total balls to overs format (like 37.4)"""
    try:
        whole_overs = balls // 6
        remaining = balls % 6
        return whole_overs + (remaining / 10)
    except:
        return 0


def get_winner_from_scores(team1, runs1, team2, runs2, overs2):
    """
    Determine match winner from scores
    Note: For DLS, if team2 overs < 50, we check if runs2 > runs1 (revised target may apply)
    """
    if runs1 is None or runs2 is None:
        return None, None, None
    
    if runs2 > runs1:
        # Team 2 won (chasing team)
        margin = 10 - (runs2 // runs1) if runs1 > 0 else 10  # Simplified wickets margin
        return team2, 'wickets', None  # Need actual wickets data
    elif runs1 > runs2:
        # Team 1 won (defending team)
        margin = runs1 - runs2
        return team1, 'runs', margin
    else:
        return 'Tie', 'tie', 0


def find_team_code(team_name):
    """Convert full team name to code"""
    team_name_lower = team_name.lower().strip()
    
    for code, name in TEAM_NAMES.items():
        if name.lower() == team_name_lower or code.lower() == team_name_lower:
            return code
    
    # Partial matches
    for code, name in TEAM_NAMES.items():
        if team_name_lower in name.lower() or name.lower() in team_name_lower:
            return code
    
    return team_name  # Return original if no match


# =============================================================================
# MAIN PARSING FUNCTIONS
# =============================================================================

def parse_single_match(match_folder, is_plate=False):
    """
    Parse a single match folder containing meta.csv, batting.csv, bowling.csv
    Returns: match_data (dict), batting_df, bowling_df
    """
    match_data = {
        'match_id': os.path.basename(match_folder),
        'is_plate': is_plate
    }
    
    # Parse meta.csv
    meta_path = os.path.join(match_folder, 'meta.csv')
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        if len(lines) >= 6:
            # Line 0: Index (ignore)
            # Line 1: Team 1 score
            # Line 2: Team 2 score
            # Line 3: Venue
            # Line 4: Date
            # Line 5: Toss info
            
            team1, runs1, wickets1, overs1 = parse_score(lines[1])
            team2, runs2, wickets2, overs2 = parse_score(lines[2])
            
            venue = lines[3].strip().strip('"')
            
            day_name, month, day_num, stage = parse_date(lines[4])
            
            toss_winner_name, toss_decision = parse_toss_info(lines[5])
            
            # Determine toss winner code
            toss_winner = None
            if toss_winner_name:
                toss_winner = find_team_code(toss_winner_name)
                # Verify against teams
                if toss_winner not in [team1, team2]:
                    # Try matching
                    if team1 and toss_winner_name.lower() in TEAM_NAMES.get(team1, '').lower():
                        toss_winner = team1
                    elif team2 and toss_winner_name.lower() in TEAM_NAMES.get(team2, '').lower():
                        toss_winner = team2
            
            # Determine batting first team
            if toss_decision == 'Bat':
                batting_first = toss_winner
            elif toss_decision == 'Bowl':
                batting_first = team2 if toss_winner == team1 else team1
            else:
                batting_first = team1  # Default assumption
            
            # Ensure team1 is always the team that batted first
            if batting_first == team2:
                # Swap
                team1, team2 = team2, team1
                runs1, runs2 = runs2, runs1
                wickets1, wickets2 = wickets2, wickets1
                overs1, overs2 = overs2, overs1
            
            # Determine winner
            if runs2 is not None and runs1 is not None:
                if runs2 > runs1:
                    winner = team2
                    win_type = 'wickets'
                    win_margin = 10 - wickets2 if wickets2 else None
                elif runs1 > runs2:
                    winner = team1
                    win_type = 'runs'
                    win_margin = runs1 - runs2
                else:
                    winner = 'Tie'
                    win_type = 'tie'
                    win_margin = 0
            else:
                winner = None
                win_type = None
                win_margin = None
            
            match_data.update({
                'team1': team1,
                'team2': team2,
                'team1_runs': runs1,
                'team1_wickets': wickets1,
                'team1_overs': overs1,
                'team2_runs': runs2,
                'team2_wickets': wickets2,
                'team2_overs': overs2,
                'venue': venue,
                'day_name': day_name,
                'month': month,
                'day_num': day_num,
                'stage': stage,
                'toss_winner': toss_winner,
                'toss_decision': toss_decision,
                'batting_first': team1,  # After swap, team1 always bats first
                'winner': winner,
                'win_type': win_type,
                'win_margin': win_margin,
                'total_runs': (runs1 or 0) + (runs2 or 0),
                'total_wickets': (wickets1 or 0) + (wickets2 or 0)
            })
    except Exception as e:
        print(f"Error parsing meta for {match_folder}: {e}")
    
    # Parse batting.csv
    batting_df = pd.DataFrame()
    batting_path = os.path.join(match_folder, 'batting.csv')
    try:
        batting_df = pd.read_csv(batting_path)
        batting_df['match_id'] = match_data['match_id']
        batting_df['is_plate'] = is_plate
        
        # Clean column names
        batting_df.columns = [col.strip() for col in batting_df.columns]
        
        # Add dismissal type
        if 'Dismissal' in batting_df.columns:
            batting_df['dismissal_type'] = batting_df['Dismissal'].apply(parse_dismissal)
        
        # Add batting position
        batting_df['batting_position'] = batting_df.groupby('Team').cumcount() + 1
        
        # Clean numeric columns
        for col in ['Runs', 'Balls', '4s', '6s', 'SR']:
            if col in batting_df.columns:
                batting_df[col] = pd.to_numeric(batting_df[col], errors='coerce').fillna(0)
        
    except Exception as e:
        print(f"Error parsing batting for {match_folder}: {e}")
    
    # Parse bowling.csv
    bowling_df = pd.DataFrame()
    bowling_path = os.path.join(match_folder, 'bowling.csv')
    try:
        bowling_df = pd.read_csv(bowling_path)
        bowling_df['match_id'] = match_data['match_id']
        bowling_df['is_plate'] = is_plate
        
        # Clean column names
        bowling_df.columns = [col.strip() for col in bowling_df.columns]
        
        # Clean numeric columns
        for col in ['Overs', 'Maidens', 'Runs', 'Wickets', 'Economy']:
            if col in bowling_df.columns:
                bowling_df[col] = pd.to_numeric(bowling_df[col], errors='coerce').fillna(0)
        
        # Add bowling position (order)
        bowling_df['bowling_position'] = bowling_df.groupby('Team').cumcount() + 1
        
    except Exception as e:
        print(f"Error parsing bowling for {match_folder}: {e}")
    
    return match_data, batting_df, bowling_df


def parse_all_matches(raw_path, plate_path):
    """
    Parse all matches from both main and plate data folders
    Returns: matches_df, all_batting_df, all_bowling_df
    """
    all_matches = []
    all_batting = []
    all_bowling = []
    
    # Parse main data
    print(f"Parsing main matches from {raw_path}...")
    main_folders = [f for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f))]
    
    for folder in sorted(main_folders):
        folder_path = os.path.join(raw_path, folder)
        match_data, batting_df, bowling_df = parse_single_match(folder_path, is_plate=False)
        all_matches.append(match_data)
        if not batting_df.empty:
            all_batting.append(batting_df)
        if not bowling_df.empty:
            all_bowling.append(bowling_df)
    
    print(f"  Parsed {len(main_folders)} main matches")
    
    # Parse plate data
    print(f"Parsing plate matches from {plate_path}...")
    plate_folders = [f for f in os.listdir(plate_path) 
                     if os.path.isdir(os.path.join(plate_path, f))]
    
    for folder in sorted(plate_folders):
        folder_path = os.path.join(plate_path, folder)
        match_data, batting_df, bowling_df = parse_single_match(folder_path, is_plate=True)
        match_data['stage'] = 'Plate Group'  # Override stage for plate matches
        all_matches.append(match_data)
        if not batting_df.empty:
            all_batting.append(batting_df)
        if not bowling_df.empty:
            all_bowling.append(bowling_df)
    
    print(f"  Parsed {len(plate_folders)} plate matches")
    
    # Combine into DataFrames
    matches_df = pd.DataFrame(all_matches)
    batting_df = pd.concat(all_batting, ignore_index=True) if all_batting else pd.DataFrame()
    bowling_df = pd.concat(all_bowling, ignore_index=True) if all_bowling else pd.DataFrame()
    
    # Add group information to matches
    matches_df['team1_group'] = matches_df['team1'].apply(get_team_group)
    matches_df['team2_group'] = matches_df['team2'].apply(get_team_group)
    
    print(f"\nTotal: {len(matches_df)} matches, {len(batting_df)} batting records, {len(bowling_df)} bowling records")
    
    return matches_df, batting_df, bowling_df


def parse_standings(raw_path, plate_path):
    """Parse standings files"""
    
    # Main standings
    main_standings_path = os.path.join(raw_path, 'standings_collective.csv')
    main_standings = pd.read_csv(main_standings_path)
    
    # Clean team names (remove Q, E markers)
    main_standings['Team_Clean'] = main_standings['Team'].str.replace(r'\s*\([QE]\)', '', regex=True)
    main_standings['Qualified'] = main_standings['Team'].str.contains(r'\(Q\)', regex=True)
    
    # Plate standings
    plate_standings_path = os.path.join(plate_path, 'standings.csv.txt')
    plate_standings = pd.read_csv(plate_standings_path)
    plate_standings['Group'] = 'Plate'
    
    # Rename columns to match
    plate_standings = plate_standings.rename(columns={
        'Position': 'Pos',
        'Matches': 'P',
        'Wins': 'W',
        'Losses': 'L',
        'Points': 'Pts'
    })
    
    plate_standings['Team_Clean'] = plate_standings['Team'].str.replace(r'\s*\([QE]\)', '', regex=True)
    plate_standings['Qualified'] = plate_standings['Team'].str.contains(r'\(Q\)', regex=True)
    
    # Combine
    standings = pd.concat([main_standings, plate_standings], ignore_index=True)
    
    return standings


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_team_batting_stats(batting_df, matches_df):
    """Calculate all batting statistics per team"""
    
    teams_batting = []
    
    for team in batting_df['Team'].unique():
        team_data = batting_df[batting_df['Team'] == team]
        
        # Get matches played by this team
        team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
        matches_played = len(team_matches)
        
        # Basic stats
        total_runs = team_data['Runs'].sum()
        total_balls = team_data['Balls'].sum()
        total_4s = team_data['4s'].sum()
        total_6s = team_data['6s'].sum()
        
        # Innings count (team innings, not player)
        innings = team_data['match_id'].nunique()
        
        # Average score per match
        avg_score = total_runs / innings if innings > 0 else 0
        
        # Team strike rate
        team_sr = (total_runs / total_balls * 100) if total_balls > 0 else 0
        
        # Boundary percentage
        boundary_runs = (total_4s * 4) + (total_6s * 6)
        boundary_pct = (boundary_runs / total_runs * 100) if total_runs > 0 else 0
        
        # 50s and 100s count
        fifties = len(team_data[(team_data['Runs'] >= 50) & (team_data['Runs'] < 100)])
        hundreds = len(team_data[team_data['Runs'] >= 100])
        
        # Top 3 batters dependency
        top3_runs = team_data[team_data['batting_position'] <= 3]['Runs'].sum()
        top3_dependency = (top3_runs / total_runs * 100) if total_runs > 0 else 0
        
        # Opening partnership value (positions 1 and 2)
        openers_runs = team_data[team_data['batting_position'] <= 2]['Runs'].sum()
        openers_avg = openers_runs / innings if innings > 0 else 0
        
        # Finisher rating (positions 5-7 with SR > 100)
        finishers = team_data[(team_data['batting_position'] >= 5) & 
                               (team_data['batting_position'] <= 7) &
                               (team_data['SR'] > 100)]
        finisher_runs = finishers['Runs'].sum()
        
        # Middle order contribution (positions 4-6)
        middle_order = team_data[(team_data['batting_position'] >= 4) & 
                                  (team_data['batting_position'] <= 6)]
        middle_order_runs = middle_order['Runs'].sum()
        middle_order_pct = (middle_order_runs / total_runs * 100) if total_runs > 0 else 0
        
        # Get team scores for consistency metrics
        team_scores = []
        for _, match in team_matches.iterrows():
            if match['team1'] == team:
                team_scores.append(match['team1_runs'])
            else:
                team_scores.append(match['team2_runs'])
        
        team_scores = [s for s in team_scores if s is not None and not pd.isna(s)]
        
        # Consistency score (250+ frequency)
        scores_250_plus = len([s for s in team_scores if s >= 250])
        consistency_score = (scores_250_plus / len(team_scores) * 100) if team_scores else 0
        
        # Highest and lowest scores
        highest_score = max(team_scores) if team_scores else 0
        lowest_score = min(team_scores) if team_scores else 0
        
        # Dismissal analysis (how team batters got out)
        dismissal_counts = team_data['dismissal_type'].value_counts().to_dict()
        total_dismissals = sum([v for k, v in dismissal_counts.items() if k not in ['not out', 'batting', 'unknown']])
        
        caught_pct = (dismissal_counts.get('caught', 0) / total_dismissals * 100) if total_dismissals > 0 else 0
        bowled_pct = (dismissal_counts.get('bowled', 0) / total_dismissals * 100) if total_dismissals > 0 else 0
        lbw_pct = (dismissal_counts.get('lbw', 0) / total_dismissals * 100) if total_dismissals > 0 else 0
        
        teams_batting.append({
            'team': team,
            'team_name': TEAM_NAMES.get(team, team),
            'group': get_team_group(team),
            'matches_played': matches_played,
            'innings_batted': innings,
            'total_runs': total_runs,
            'total_balls': total_balls,
            'total_4s': total_4s,
            'total_6s': total_6s,
            'avg_score': round(avg_score, 2),
            'team_sr': round(team_sr, 2),
            'boundary_runs': boundary_runs,
            'boundary_pct': round(boundary_pct, 2),
            'fifties': fifties,
            'hundreds': hundreds,
            'top3_runs': top3_runs,
            'top3_dependency': round(top3_dependency, 2),
            'openers_avg': round(openers_avg, 2),
            'middle_order_pct': round(middle_order_pct, 2),
            'finisher_runs': finisher_runs,
            'consistency_score': round(consistency_score, 2),
            'highest_score': highest_score,
            'lowest_score': lowest_score,
            'caught_pct': round(caught_pct, 2),
            'bowled_pct': round(bowled_pct, 2),
            'lbw_pct': round(lbw_pct, 2)
        })
    
    return pd.DataFrame(teams_batting)


def calculate_team_bowling_stats(bowling_df, matches_df):
    """Calculate all bowling statistics per team"""
    
    teams_bowling = []
    
    for team in bowling_df['Team'].unique():
        team_data = bowling_df[bowling_df['Team'] == team]
        
        # Get matches played by this team
        team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
        matches_played = len(team_matches)
        
        # Basic stats
        total_overs = team_data['Overs'].sum()
        total_maidens = team_data['Maidens'].sum()
        total_runs_conceded = team_data['Runs'].sum()
        total_wickets = team_data['Wickets'].sum()
        
        # Innings count
        innings = team_data['match_id'].nunique()
        
        # Bowling average
        bowling_avg = total_runs_conceded / total_wickets if total_wickets > 0 else 999
        
        # Economy rate
        economy = total_runs_conceded / total_overs if total_overs > 0 else 0
        
        # Bowling strike rate (balls per wicket)
        total_balls_bowled = sum([overs_to_balls(ov) for ov in team_data['Overs']])
        bowling_sr = total_balls_bowled / total_wickets if total_wickets > 0 else 999
        
        # Wickets per match
        wickets_per_match = total_wickets / innings if innings > 0 else 0
        
        # Maidens per match
        maidens_per_match = total_maidens / innings if innings > 0 else 0
        
        # 5-wicket hauls
        five_wkt_hauls = len(team_data[team_data['Wickets'] >= 5])
        
        # 3+ wicket hauls
        three_plus_wkt_hauls = len(team_data[team_data['Wickets'] >= 3])
        
        # Best bowling figures
        best_bowling = team_data.loc[team_data['Wickets'].idxmax()] if len(team_data) > 0 else None
        best_bowling_fig = f"{int(best_bowling['Wickets'])}/{int(best_bowling['Runs'])}" if best_bowling is not None else "0/0"
        
        # Last 3 bowlers economy (death bowling proxy)
        # Get last 3 bowlers per innings
        last_bowlers_econ = []
        for match_id in team_data['match_id'].unique():
            match_bowling = team_data[team_data['match_id'] == match_id].sort_values('bowling_position', ascending=False)
            last_3 = match_bowling.head(3)
            if not last_3.empty:
                innings_last3_runs = last_3['Runs'].sum()
                innings_last3_overs = last_3['Overs'].sum()
                if innings_last3_overs > 0:
                    last_bowlers_econ.append(innings_last3_runs / innings_last3_overs)
        
        death_bowling_economy = np.mean(last_bowlers_econ) if last_bowlers_econ else 0
        
        # Runs conceded per match
        runs_conceded_per_match = total_runs_conceded / innings if innings > 0 else 0
        
        teams_bowling.append({
            'team': team,
            'team_name': TEAM_NAMES.get(team, team),
            'group': get_team_group(team),
            'matches_played': matches_played,
            'innings_bowled': innings,
            'total_overs': round(total_overs, 1),
            'total_balls_bowled': total_balls_bowled,
            'total_maidens': total_maidens,
            'total_runs_conceded': total_runs_conceded,
            'total_wickets': total_wickets,
            'bowling_avg': round(bowling_avg, 2),
            'economy': round(economy, 2),
            'bowling_sr': round(bowling_sr, 2),
            'wickets_per_match': round(wickets_per_match, 2),
            'maidens_per_match': round(maidens_per_match, 2),
            'five_wkt_hauls': five_wkt_hauls,
            'three_plus_wkt_hauls': three_plus_wkt_hauls,
            'best_bowling': best_bowling_fig,
            'death_bowling_economy': round(death_bowling_economy, 2),
            'runs_conceded_per_match': round(runs_conceded_per_match, 2)
        })
    
    return pd.DataFrame(teams_bowling)


def calculate_team_match_stats(matches_df, standings_df):
    """Calculate match-related statistics per team"""
    
    teams_match = []
    
    # Get all unique teams
    all_teams = set(matches_df['team1'].dropna().unique()) | set(matches_df['team2'].dropna().unique())
    
    for team in all_teams:
        team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
        
        matches_played = len(team_matches)
        wins = len(team_matches[team_matches['winner'] == team])
        losses = matches_played - wins
        
        win_pct = (wins / matches_played * 100) if matches_played > 0 else 0
        
        # Toss analysis
        toss_wins = len(team_matches[team_matches['toss_winner'] == team])
        toss_win_pct = (toss_wins / matches_played * 100) if matches_played > 0 else 0
        
        # When won toss, what was the decision?
        toss_won_matches = team_matches[team_matches['toss_winner'] == team]
        toss_bat_count = len(toss_won_matches[toss_won_matches['toss_decision'] == 'Bat'])
        toss_bowl_count = len(toss_won_matches[toss_won_matches['toss_decision'] == 'Bowl'])
        
        # Win after winning toss
        toss_won_and_match_won = len(toss_won_matches[toss_won_matches['winner'] == team])
        toss_win_to_match_win_pct = (toss_won_and_match_won / toss_wins * 100) if toss_wins > 0 else 0
        
        # Batting first vs second analysis
        bat_first_matches = team_matches[team_matches['batting_first'] == team]
        bat_first_wins = len(bat_first_matches[bat_first_matches['winner'] == team])
        bat_first_win_pct = (bat_first_wins / len(bat_first_matches) * 100) if len(bat_first_matches) > 0 else 0
        
        bat_second_matches = team_matches[team_matches['batting_first'] != team]
        bat_second_wins = len(bat_second_matches[bat_second_matches['winner'] == team])
        bat_second_win_pct = (bat_second_wins / len(bat_second_matches) * 100) if len(bat_second_matches) > 0 else 0
        
        # Win margin analysis
        team_wins = team_matches[team_matches['winner'] == team]
        
        # Pressure wins (< 20 runs or < 3 wickets)
        runs_wins = team_wins[team_wins['win_type'] == 'runs']
        close_runs_wins = len(runs_wins[runs_wins['win_margin'] < 20])
        
        wickets_wins = team_wins[team_wins['win_type'] == 'wickets']
        close_wickets_wins = len(wickets_wins[wickets_wins['win_margin'] < 3]) if not wickets_wins.empty else 0
        
        pressure_wins = close_runs_wins + close_wickets_wins
        
        # Dominant wins (80+ runs or 7+ wickets)
        big_runs_wins = len(runs_wins[runs_wins['win_margin'] >= 80]) if not runs_wins.empty else 0
        big_wickets_wins = len(wickets_wins[wickets_wins['win_margin'] >= 7]) if not wickets_wins.empty else 0
        dominant_wins = big_runs_wins + big_wickets_wins
        
        # Get NRR from standings
        team_standing = standings_df[standings_df['Team_Clean'] == team]
        nrr = team_standing['NRR'].values[0] if len(team_standing) > 0 else 0
        points = team_standing['Pts'].values[0] if len(team_standing) > 0 else 0
        qualified = team_standing['Qualified'].values[0] if len(team_standing) > 0 else False
        
        # Chase analysis
        chases = bat_second_matches  # Matches where team batted second
        total_chases = len(chases)
        successful_chases = len(chases[chases['winner'] == team])
        chase_success_rate = (successful_chases / total_chases * 100) if total_chases > 0 else 0
        
        # Big chase success (280+)
        big_chases = chases[chases['team1_runs'] >= 280]  # team1 is batting first
        big_chase_success = len(big_chases[big_chases['winner'] == team])
        big_chase_attempts = len(big_chases)
        
        # Defend analysis
        defends = bat_first_matches
        total_defends = len(defends)
        successful_defends = len(defends[defends['winner'] == team])
        defend_success_rate = (successful_defends / total_defends * 100) if total_defends > 0 else 0
        
        teams_match.append({
            'team': team,
            'team_name': TEAM_NAMES.get(team, team),
            'group': get_team_group(team),
            'matches_played': matches_played,
            'wins': wins,
            'losses': losses,
            'win_pct': round(win_pct, 2),
            'points': points,
            'nrr': nrr,
            'qualified': qualified,
            'toss_wins': toss_wins,
            'toss_win_pct': round(toss_win_pct, 2),
            'toss_bat_decisions': toss_bat_count,
            'toss_bowl_decisions': toss_bowl_count,
            'toss_win_to_match_win_pct': round(toss_win_to_match_win_pct, 2),
            'bat_first_matches': len(bat_first_matches),
            'bat_first_wins': bat_first_wins,
            'bat_first_win_pct': round(bat_first_win_pct, 2),
            'bat_second_matches': len(bat_second_matches),
            'bat_second_wins': bat_second_wins,
            'bat_second_win_pct': round(bat_second_win_pct, 2),
            'chase_success_rate': round(chase_success_rate, 2),
            'defend_success_rate': round(defend_success_rate, 2),
            'big_chase_attempts': big_chase_attempts,
            'big_chase_success': big_chase_success,
            'pressure_wins': pressure_wins,
            'dominant_wins': dominant_wins
        })
    
    return pd.DataFrame(teams_match)


def calculate_dismissal_stats(batting_df):
    """Calculate dismissal statistics for teams (both as batting and fielding sides)"""
    
    # Dismissal types by batting team (how they got out)
    batting_dismissals = batting_df.groupby(['Team', 'dismissal_type']).size().unstack(fill_value=0)
    batting_dismissals = batting_dismissals.reset_index()
    batting_dismissals['total_dismissals'] = batting_dismissals.select_dtypes(include=[np.number]).sum(axis=1)
    
    # Calculate by opposition (fielding effectiveness)
    # This requires mapping batting team to opposition from matches
    # For simplicity, we'll use the main dismissal stats
    
    return batting_dismissals


def calculate_venue_stats(matches_df):
    """Calculate venue-wise statistics"""
    
    venue_stats = []
    
    for venue in matches_df['venue'].dropna().unique():
        venue_matches = matches_df[matches_df['venue'] == venue]
        
        total_matches = len(venue_matches)
        
        # Average first innings score
        avg_first_innings = venue_matches['team1_runs'].mean()
        
        # Average second innings score
        avg_second_innings = venue_matches['team2_runs'].mean()
        
        # Chase success rate
        chases = len(venue_matches[venue_matches['winner'] == venue_matches['team2']])
        chase_success_rate = (chases / total_matches * 100) if total_matches > 0 else 0
        
        # Defend success rate
        defends = total_matches - chases
        defend_success_rate = (defends / total_matches * 100) if total_matches > 0 else 0
        
        # Toss decision preference
        bat_decisions = len(venue_matches[venue_matches['toss_decision'] == 'Bat'])
        bowl_decisions = len(venue_matches[venue_matches['toss_decision'] == 'Bowl'])
        
        # High-scoring matches (300+)
        high_scoring = len(venue_matches[(venue_matches['team1_runs'] >= 300) | (venue_matches['team2_runs'] >= 300)])
        
        venue_stats.append({
            'venue': venue,
            'total_matches': total_matches,
            'avg_first_innings': round(avg_first_innings, 2) if not pd.isna(avg_first_innings) else 0,
            'avg_second_innings': round(avg_second_innings, 2) if not pd.isna(avg_second_innings) else 0,
            'chase_wins': chases,
            'defend_wins': defends,
            'chase_success_rate': round(chase_success_rate, 2),
            'defend_success_rate': round(defend_success_rate, 2),
            'toss_bat_decisions': bat_decisions,
            'toss_bowl_decisions': bowl_decisions,
            'high_scoring_matches': high_scoring
        })
    
    return pd.DataFrame(venue_stats)


def merge_team_stats(team_batting, team_bowling, team_match):
    """Merge all team statistics into one comprehensive DataFrame"""
    
    # Merge batting and bowling
    team_stats = pd.merge(
        team_match,
        team_batting.drop(columns=['team_name', 'group', 'matches_played'], errors='ignore'),
        on='team',
        how='outer'
    )
    
    team_stats = pd.merge(
        team_stats,
        team_bowling.drop(columns=['team_name', 'group', 'matches_played'], errors='ignore'),
        on='team',
        how='outer'
    )
    
    # Calculate combined metrics
    
    # Balanced team index (lower is better - means good at both)
    # Rank teams by batting avg score (desc) and bowling economy (asc)
    team_stats['batting_rank'] = team_stats['avg_score'].rank(ascending=False)
    team_stats['bowling_rank'] = team_stats['economy'].rank(ascending=True)
    team_stats['balanced_index'] = (team_stats['batting_rank'] + team_stats['bowling_rank']) / 2
    
    # Overall team rating (custom metric)
    # Combines win%, NRR, avg score, economy
    team_stats['team_rating'] = (
        team_stats['win_pct'] * 0.4 +
        (team_stats['nrr'] + 5) * 10 * 0.2 +  # Normalize NRR
        team_stats['avg_score'] / 3 * 0.2 +
        (15 - team_stats['economy']) * 10 * 0.2  # Invert economy
    )
    
    return team_stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("VIJAY HAZARE TROPHY 2024-25 DATA PROCESSOR")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Step 1: Parse all matches
    print("\n[Step 1/6] Parsing all matches...")
    matches_df, batting_df, bowling_df = parse_all_matches(RAW_DATA_PATH, PLATE_DATA_PATH)
    
    # Step 2: Parse standings
    print("\n[Step 2/6] Parsing standings...")
    standings_df = parse_standings(RAW_DATA_PATH, PLATE_DATA_PATH)
    print(f"  Loaded standings for {len(standings_df)} teams")
    
    # Step 3: Calculate team batting stats
    print("\n[Step 3/6] Calculating team batting statistics...")
    team_batting = calculate_team_batting_stats(batting_df, matches_df)
    print(f"  Calculated batting stats for {len(team_batting)} teams")
    
    # Step 4: Calculate team bowling stats
    print("\n[Step 4/6] Calculating team bowling statistics...")
    team_bowling = calculate_team_bowling_stats(bowling_df, matches_df)
    print(f"  Calculated bowling stats for {len(team_bowling)} teams")
    
    # Step 5: Calculate team match stats
    print("\n[Step 5/6] Calculating team match statistics...")
    team_match = calculate_team_match_stats(matches_df, standings_df)
    print(f"  Calculated match stats for {len(team_match)} teams")
    
    # Step 6: Calculate venue stats
    print("\n[Step 6/6] Calculating venue statistics...")
    venue_stats = calculate_venue_stats(matches_df)
    print(f"  Calculated stats for {len(venue_stats)} venues")
    
    # Merge all team stats
    print("\n[Merging] Combining all team statistics...")
    team_stats = merge_team_stats(team_batting, team_bowling, team_match)
    
    # Save all files
    print("\n[Saving] Writing output files...")
    
    # 1. Matches
    matches_df.to_csv(os.path.join(OUTPUT_PATH, 'matches.csv'), index=False)
    print(f"  ✓ matches.csv ({len(matches_df)} records)")
    
    # 2. Team stats (comprehensive)
    team_stats.to_csv(os.path.join(OUTPUT_PATH, 'team_stats.csv'), index=False)
    print(f"  ✓ team_stats.csv ({len(team_stats)} records)")
    
    # 3. Player batting
    batting_df.to_csv(os.path.join(OUTPUT_PATH, 'player_batting.csv'), index=False)
    print(f"  ✓ player_batting.csv ({len(batting_df)} records)")
    
    # 4. Player bowling
    bowling_df.to_csv(os.path.join(OUTPUT_PATH, 'player_bowling.csv'), index=False)
    print(f"  ✓ player_bowling.csv ({len(bowling_df)} records)")
    
    # 5. Venue stats
    venue_stats.to_csv(os.path.join(OUTPUT_PATH, 'venue_stats.csv'), index=False)
    print(f"  ✓ venue_stats.csv ({len(venue_stats)} records)")
    
    # 6. Standings
    standings_df.to_csv(os.path.join(OUTPUT_PATH, 'standings.csv'), index=False)
    print(f"  ✓ standings.csv ({len(standings_df)} records)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput files created in '{OUTPUT_PATH}/' folder:")
    print(f"  1. matches.csv        - All {len(matches_df)} match details")
    print(f"  2. team_stats.csv     - All {len(team_stats)} team statistics (comprehensive)")
    print(f"  3. player_batting.csv - All {len(batting_df)} player batting records")
    print(f"  4. player_bowling.csv - All {len(bowling_df)} player bowling records")
    print(f"  5. venue_stats.csv    - All {len(venue_stats)} venue statistics")
    print(f"  6. standings.csv      - All {len(standings_df)} team standings")
    
    print("\n" + "-" * 60)
    print("KEY STATISTICS SUMMARY")
    print("-" * 60)
    
    # Tournament totals
    total_runs = matches_df['total_runs'].sum()
    total_wickets = matches_df['total_wickets'].sum()
    highest_score = max(matches_df['team1_runs'].max(), matches_df['team2_runs'].max())
    lowest_score = min(matches_df[matches_df['team1_runs'] > 0]['team1_runs'].min(), 
                       matches_df[matches_df['team2_runs'] > 0]['team2_runs'].min())
    
    print(f"  Total Matches: {len(matches_df)}")
    print(f"  Total Runs Scored: {int(total_runs)}")
    print(f"  Total Wickets Fallen: {int(total_wickets)}")
    print(f"  Highest Team Score: {int(highest_score)}")
    print(f"  Lowest Team Score: {int(lowest_score)}")
    
    # Top teams
    print(f"\n  Top 5 Teams by Win %:")
    top_teams = team_stats.nlargest(5, 'win_pct')[['team', 'team_name', 'win_pct', 'nrr']]
    for _, row in top_teams.iterrows():
        print(f"    {row['team']:6} - {row['win_pct']:.1f}% (NRR: {row['nrr']})")
    
    # Champion
    print(f"\n  🏆 CHAMPION: VIDARBHA")
    vidarbha = team_stats[team_stats['team'] == 'VID']
    if not vidarbha.empty:
        v = vidarbha.iloc[0]
        print(f"     Matches: {int(v['matches_played'])}, Wins: {int(v['wins'])}, Win%: {v['win_pct']:.1f}%")
        print(f"     Avg Score: {v['avg_score']:.1f}, Economy: {v['economy']:.2f}")
    
    print("\n" + "=" * 60)
    print("Ready for dashboard development!")
    print("=" * 60)
    
    return matches_df, team_stats, batting_df, bowling_df, venue_stats, standings_df


if __name__ == "__main__":
    main()