"""
Vijay Hazare Trophy 2025-26 Dashboard
=====================================
Complete 9-page interactive dashboard built with Dash and Plotly.

Pages:
1. Tournament Overview
2. Team Comparison
3. Batting Analysis
4. Bowling Analysis
5. Toss & Venue Impact
6. Match Situations
7. Dismissal Patterns
8. Qualified Teams Deep Dive
9. Champion's Journey (Vidarbha)

Run: python app.py
Access: http://127.0.0.1:8050
"""

# =============================================================================
# IMPORTS
# =============================================================================

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

# =============================================================================
# LOAD DATA
# =============================================================================

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load all CSV files with absolute paths
matches = pd.read_csv(os.path.join(DATA_DIR, 'matches.csv'))
team_stats = pd.read_csv(os.path.join(DATA_DIR, 'team_stats.csv'))
player_batting = pd.read_csv(os.path.join(DATA_DIR, 'player_batting.csv'))
player_bowling = pd.read_csv(os.path.join(DATA_DIR, 'player_bowling.csv'))
venue_stats = pd.read_csv(os.path.join(DATA_DIR, 'venue_stats.csv'))
standings = pd.read_csv(os.path.join(DATA_DIR, 'standings.csv'))

# =============================================================================
# CONSTANTS
# =============================================================================

# Color Palette
COLORS = {
    'background': '#f8f9fa',
    'card': '#ffffff',
    'primary': '#e63946',
    'secondary': '#1d3557',
    'tertiary': '#457b9d',
    'success': '#2a9d8f',
    'warning': '#f4a261',
    'gold': '#ffb703',
    'purple': '#6a4c93',
    'text': '#212529',
    'text_muted': '#6c757d',
    'border': '#dee2e6'
}

# Team Colors for charts
TEAM_COLORS = {
    'VID': '#e63946', 'SAUR': '#1d3557', 'DEL': '#457b9d', 'MUM': '#2a9d8f',
    'KAR': '#f4a261', 'UP': '#ffb703', 'PUN': '#6a4c93', 'MP': '#e76f51',
    'BIH': '#264653', 'MNP': '#2a9d8f', 'BEN': '#e9c46a', 'BRD': '#f4a261',
    'HAR': '#e63946', 'RLYS': '#1d3557', 'GUJ': '#457b9d', 'ODSA': '#2a9d8f',
    'AP': '#f4a261', 'SER': '#6c757d', 'MAH': '#ffb703', 'CG': '#6a4c93',
    'HP': '#e76f51', 'GOA': '#264653', 'UTK': '#2a9d8f', 'SKM': '#e9c46a',
    'KER': '#f4a261', 'JHKD': '#e63946', 'TN': '#1d3557', 'TRI': '#457b9d',
    'RAJ': '#2a9d8f', 'PDC': '#f4a261', 'JK': '#ffb703', 'HYD': '#6a4c93',
    'ASM': '#e76f51', 'CDG': '#264653', 'NGL': '#2a9d8f', 'MGLY': '#e9c46a',
    'ARNP': '#f4a261', 'MIZ': '#6c757d'
}

# Group Colors
GROUP_COLORS = {
    'Group A': '#e63946',
    'Group B': '#1d3557',
    'Group C': '#2a9d8f',
    'Group D': '#f4a261',
    'Plate': '#6a4c93'
}

# Team Full Names
TEAM_NAMES = {
    'AP': 'Andhra Pradesh', 'ASM': 'Assam', 'BEN': 'Bengal', 'BRD': 'Baroda',
    'CDG': 'Chandigarh', 'CG': 'Chhattisgarh', 'DEL': 'Delhi', 'GOA': 'Goa',
    'GUJ': 'Gujarat', 'HAR': 'Haryana', 'HP': 'Himachal Pradesh', 'HYD': 'Hyderabad',
    'JHKD': 'Jharkhand', 'JK': 'Jammu & Kashmir', 'KAR': 'Karnataka', 'KER': 'Kerala',
    'MAH': 'Maharashtra', 'MP': 'Madhya Pradesh', 'MUM': 'Mumbai', 'ODSA': 'Odisha',
    'PDC': 'Puducherry', 'PUN': 'Punjab', 'RAJ': 'Rajasthan', 'RLYS': 'Railways',
    'SAUR': 'Saurashtra', 'SER': 'Services', 'SKM': 'Sikkim', 'TN': 'Tamil Nadu',
    'TRI': 'Tripura', 'UP': 'Uttar Pradesh', 'UTK': 'Uttarakhand', 'VID': 'Vidarbha',
    'BIH': 'Bihar', 'MNP': 'Manipur', 'NGL': 'Nagaland', 'MGLY': 'Meghalaya',
    'ARNP': 'Arunachal Pradesh', 'MIZ': 'Mizoram'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_kpi_card(title, value, subtitle=None, color=COLORS['primary'], icon=None):
    """Create a KPI card component"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H6(title, className='text-muted mb-1', style={'fontSize': '0.85rem'}),
                html.H3(value, className='mb-0', style={'color': color, 'fontWeight': 'bold'}),
                html.Small(subtitle, className='text-muted') if subtitle else None
            ])
        ])
    ], className='shadow-sm h-100', style={'borderRadius': '10px', 'border': 'none'})


def create_stat_card(title, stats_list, color=COLORS['secondary']):
    """Create a statistics card with multiple values"""
    items = []
    for label, value in stats_list:
        items.append(
            html.Div([
                html.Span(label, className='text-muted', style={'fontSize': '0.85rem'}),
                html.Span(value, className='float-end fw-bold', style={'color': color})
            ], className='mb-2')
        )
    
    return dbc.Card([
        dbc.CardHeader(title, style={'backgroundColor': color, 'color': 'white', 'fontWeight': 'bold'}),
        dbc.CardBody(items)
    ], className='shadow-sm h-100', style={'borderRadius': '10px', 'border': 'none'})


def create_section_header(title, subtitle=None):
    """Create a section header"""
    return html.Div([
        html.H4(title, className='mb-1', style={'color': COLORS['secondary'], 'fontWeight': 'bold'}),
        html.P(subtitle, className='text-muted mb-3') if subtitle else None
    ], className='mb-4')


def get_team_color(team):
    """Get color for a team"""
    return TEAM_COLORS.get(team, COLORS['tertiary'])


def format_number(num):
    """Format large numbers with commas"""
    if pd.isna(num):
        return '0'
    return f'{int(num):,}'


# =============================================================================
# CALCULATE TOURNAMENT STATISTICS
# =============================================================================

# Tournament totals
total_matches = len(matches)
total_teams = team_stats['team'].nunique()
total_runs = int(player_batting['Runs'].sum())
total_wickets = int(player_bowling['Wickets'].sum())
total_4s = int(player_batting['4s'].sum())
total_6s = int(player_batting['6s'].sum())
total_centuries = len(player_batting[player_batting['Runs'] >= 100])
total_fifties = len(player_batting[(player_batting['Runs'] >= 50) & (player_batting['Runs'] < 100)])
total_5wkt_hauls = len(player_bowling[player_bowling['Wickets'] >= 5])

# Highest and lowest scores
highest_score = int(max(matches['team1_runs'].max(), matches['team2_runs'].max()))
lowest_score = int(min(matches[matches['team1_runs'] > 0]['team1_runs'].min(), 
                       matches[matches['team2_runs'] > 0]['team2_runs'].min()))

# Top performers
top_scorer = player_batting.groupby('Batter')['Runs'].sum().idxmax()
top_scorer_runs = int(player_batting.groupby('Batter')['Runs'].sum().max())
top_wicket_taker = player_bowling.groupby('Bowler')['Wickets'].sum().idxmax()
top_wicket_taker_wkts = int(player_bowling.groupby('Bowler')['Wickets'].sum().max())

# Champion details
champion = 'VID'
champion_name = 'Vidarbha'
runner_up = 'SAUR'
runner_up_name = 'Saurashtra'

# Final match details
final_match = matches[matches['stage'] == 'Final'].iloc[0]


# =============================================================================
# PAGE 1: TOURNAMENT OVERVIEW
# =============================================================================

def create_page_overview():
    """Create Tournament Overview page"""
    
    # KPI Row
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card('Total Matches', format_number(total_matches), 
                          'Group + Knockout', COLORS['primary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Teams', format_number(total_teams), 
                          '32 Main + 6 Plate', COLORS['secondary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Total Runs', format_number(total_runs), 
                          f"{format_number(total_4s)} 4s | {format_number(total_6s)} 6s", COLORS['success'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Total Wickets', format_number(total_wickets), 
                          f"{total_5wkt_hauls} Five-wicket hauls", COLORS['warning'])
        ], md=3),
    ], className='mb-4')
    
    # Champion Card
    champion_card = dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span('🏆', style={'fontSize': '3rem'}),
                html.H3('CHAMPION: VIDARBHA', className='mb-2 mt-2', 
                       style={'color': COLORS['gold'], 'fontWeight': 'bold'}),
                html.H5(f"Final: VID {int(final_match['team1_runs'])}-{int(final_match['team1_wickets'])} vs SAUR {int(final_match['team2_runs'])}-{int(final_match['team2_wickets'])}", 
                       className='text-muted'),
                html.P('Vidarbha won by 38 runs', className='mb-0')
            ], className='text-center')
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': f'2px solid {COLORS["gold"]}', 
                                           'backgroundColor': '#fffbeb'})
    
    # Tournament Records Row
    records_row = dbc.Row([
        dbc.Col([
            create_stat_card('Batting Records', [
                ('Highest Score', f'{highest_score}'),
                ('Lowest Score', f'{lowest_score}'),
                ('Total Centuries', f'{total_centuries}'),
                ('Total Fifties', f'{total_fifties}'),
                ('Top Scorer', f'{top_scorer} ({top_scorer_runs})')
            ], COLORS['primary'])
        ], md=4),
        dbc.Col([
            create_stat_card('Bowling Records', [
                ('Total Wickets', f'{format_number(total_wickets)}'),
                ('5-Wicket Hauls', f'{total_5wkt_hauls}'),
                ('Top Wicket-Taker', f'{top_wicket_taker} ({top_wicket_taker_wkts})'),
                ('Total Maidens', f'{int(player_bowling["Maidens"].sum())}'),
                ('Avg Wickets/Match', f'{total_wickets/total_matches:.1f}')
            ], COLORS['success'])
        ], md=4),
        dbc.Col([
            create_stat_card('Match Stats', [
                ('Group Stage', '112 matches'),
                ('Plate Group', '16 matches'),
                ('Quarter Finals', '4 matches'),
                ('Semi Finals', '2 matches'),
                ('Final', '1 match')
            ], COLORS['secondary'])
        ], md=4),
    ], className='mb-4')
    
    # Group Standings
    standings_section = html.Div([
        create_section_header('Group Standings', 'Final standings after group stage'),
        dbc.Tabs([
            dbc.Tab(label='Group A', tab_id='group-a'),
            dbc.Tab(label='Group B', tab_id='group-b'),
            dbc.Tab(label='Group C', tab_id='group-c'),
            dbc.Tab(label='Group D', tab_id='group-d'),
            dbc.Tab(label='Plate', tab_id='plate'),
        ], id='standings-tabs', active_tab='group-a', className='mb-3'),
        html.Div(id='standings-content')
    ], className='mb-4')
    
    # Top Teams Chart
    top_teams_by_wins = team_stats.nlargest(10, 'wins')[['team', 'team_name', 'matches_played', 'wins', 'win_pct']]
    
    fig_top_teams = px.bar(
        top_teams_by_wins,
        x='wins',
        y='team',
        orientation='h',
        color='win_pct',
        color_continuous_scale=['#fee2e2', '#ef4444', '#991b1b'],
        text='wins',
        labels={'wins': 'Wins', 'team': 'Team', 'win_pct': 'Win %'}
    )
    fig_top_teams.update_layout(
        title='Top 10 Teams by Wins',
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        showlegend=False,
        height=400
    )
    fig_top_teams.update_traces(textposition='outside')
    
    # Matches by Stage Pie
    stage_counts = matches['stage'].value_counts().reset_index()
    stage_counts.columns = ['stage', 'count']
    
    fig_stages = px.pie(
        stage_counts,
        values='count',
        names='stage',
        color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
                                  COLORS['success'], COLORS['warning']]
    )
    fig_stages.update_layout(
        title='Matches by Stage',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    
    charts_row = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_top_teams, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_stages, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=5),
    ], className='mb-4')
    
    return html.Div([
        html.H2('🏏 Tournament Overview', className='mb-4', style={'color': COLORS['secondary']}),
        kpi_row,
        champion_card,
        records_row,
        standings_section,
        charts_row
    ])


# =============================================================================
# PAGE 2: TEAM COMPARISON
# =============================================================================

def create_page_team_comparison():
    """Create Team Comparison page"""
    
    # Team selector
    team_selector = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label('Select Team for Detailed Analysis', className='fw-bold mb-2'),
                    dcc.Dropdown(
                        id='team-selector',
                        options=[{'label': f"{row['team']} - {row['team_name']}", 'value': row['team']} 
                                for _, row in team_stats.iterrows()],
                        value='VID',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], md=6),
                dbc.Col([
                    html.Label('Compare With', className='fw-bold mb-2'),
                    dcc.Dropdown(
                        id='team-compare-selector',
                        options=[{'label': f"{row['team']} - {row['team_name']}", 'value': row['team']} 
                                for _, row in team_stats.iterrows()],
                        value='SAUR',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], md=6),
            ])
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    # Batting vs Bowling Strength Scatter
    fig_quadrant = px.scatter(
        team_stats,
        x='avg_score',
        y='economy',
        size='wins',
        hover_name='team_name',
        hover_data=['wins', 'win_pct', 'nrr'],
        
        labels={'avg_score': 'Avg Score (Batting Strength)', 'economy': 'Economy Rate (Lower = Better Bowling)'}
    )
    
    # Add quadrant lines
    avg_score_median = team_stats['avg_score'].median()
    economy_median = team_stats['economy'].median()
    
    fig_quadrant.add_hline(y=economy_median, line_dash='dash', line_color='gray', opacity=0.5)
    fig_quadrant.add_vline(x=avg_score_median, line_dash='dash', line_color='gray', opacity=0.5)
    
    # Add quadrant labels
    fig_quadrant.add_annotation(x=team_stats['avg_score'].max()-10, y=team_stats['economy'].min()+0.2,
                                text="Strong Bat + Strong Bowl", showarrow=False, font=dict(size=10, color='green'))
    fig_quadrant.add_annotation(x=team_stats['avg_score'].min()+10, y=team_stats['economy'].min()+0.2,
                                text="Weak Bat + Strong Bowl", showarrow=False, font=dict(size=10, color='orange'))
    fig_quadrant.add_annotation(x=team_stats['avg_score'].max()-10, y=team_stats['economy'].max()-0.2,
                                text="Strong Bat + Weak Bowl", showarrow=False, font=dict(size=10, color='orange'))
    fig_quadrant.add_annotation(x=team_stats['avg_score'].min()+10, y=team_stats['economy'].max()-0.2,
                                text="Weak Bat + Weak Bowl", showarrow=False, font=dict(size=10, color='red'))
    
    fig_quadrant.update_layout(
        title='Team Strength Quadrant: Batting vs Bowling',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=500,
        yaxis={'autorange': 'reversed'}  # Lower economy is better, so reverse
    )
    
    quadrant_card = dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig_quadrant, config={'displayModeBar': False})
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    # Radar Chart placeholder (will be updated by callback)
    radar_card = dbc.Card([
        dbc.CardBody([
            dcc.Graph(id='team-radar-chart', config={'displayModeBar': False})
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    # Win % Ranking
    win_pct_sorted = team_stats.sort_values('win_pct', ascending=True)
    
    fig_win_pct = px.bar(
        win_pct_sorted,
        x='win_pct',
        y='team',
        orientation='h',
        text=win_pct_sorted['win_pct'].apply(lambda x: f'{x:.1f}%'),
        labels={'win_pct': 'Win Percentage', 'team': 'Team', 'qualified': 'Qualified'}
    )
    fig_win_pct.update_layout(
        title='Win Percentage Ranking (All Teams)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=700,
        showlegend=True,
        legend_title='Qualified for Knockouts'
    )
    fig_win_pct.update_traces(textposition='outside')
    
    # NRR Ranking
    nrr_sorted = team_stats.sort_values('nrr', ascending=True)
    
    fig_nrr = px.bar(
        nrr_sorted,
        x='nrr',
        y='team',
        orientation='h',
        text=nrr_sorted['nrr'].apply(lambda x: f'{x:+.3f}'),
        labels={'nrr': 'Net Run Rate', 'team': 'Team'}
    )
    fig_nrr.update_layout(
        title='Net Run Rate Ranking',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=700,
        showlegend=True,
        legend_title='NRR'
    )
    fig_nrr.update_traces(textposition='outside')
    
    rankings_row = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_win_pct, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_nrr, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Team comparison stats (will be updated by callback)
    comparison_stats = html.Div(id='team-comparison-stats', className='mb-4')
    
    return html.Div([
        html.H2('📊 Team Comparison', className='mb-4', style={'color': COLORS['secondary']}),
        team_selector,
        comparison_stats,
        dbc.Row([
            dbc.Col([quadrant_card], md=7),
            dbc.Col([radar_card], md=5),
        ]),
        create_section_header('Team Rankings', 'Win percentage and Net Run Rate comparison'),
        rankings_row
    ])


# =============================================================================
# PAGE 3: BATTING ANALYSIS
# =============================================================================

def create_page_batting():
    """Create Batting Analysis page"""
    
    # Top KPIs
    avg_team_score = team_stats['avg_score'].mean()
    avg_team_sr = team_stats['team_sr'].mean()
    avg_boundary_pct = team_stats['boundary_pct'].mean()
    
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card('Total Runs', format_number(total_runs), 
                          f'Avg: {avg_team_score:.1f}/match', COLORS['primary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Centuries', str(total_centuries), 
                          f'{total_fifties} Fifties', COLORS['gold'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Avg Strike Rate', f'{avg_team_sr:.1f}', 
                          'Team average', COLORS['success'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Boundary %', f'{avg_boundary_pct:.1f}%', 
                          f"{format_number(total_4s)} 4s | {format_number(total_6s)} 6s", COLORS['warning'])
        ], md=3),
    ], className='mb-4')
    
    # Top 10 Teams by Total Runs
    top_batting_teams = team_stats.nlargest(10, 'total_runs')[['team', 'team_name', 'total_runs', 'avg_score', 'team_sr']]
    
    fig_total_runs = px.bar(
        top_batting_teams,
        x='total_runs',
        y='team',
        orientation='h',
        color='avg_score',
        color_continuous_scale='Reds',
        text='total_runs',
        labels={'total_runs': 'Total Runs', 'team': 'Team', 'avg_score': 'Avg Score'}
    )
    fig_total_runs.update_layout(
        title='Top 10 Teams by Total Runs',
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_total_runs.update_traces(textposition='outside')
    
    # Avg Score vs Strike Rate Scatter
    fig_score_sr = px.scatter(
        team_stats,
        x='avg_score',
        y='team_sr',
        size='wins',
        hover_name='team_name',
        hover_data=['total_runs', 'boundary_pct'],
        
        labels={'avg_score': 'Average Score', 'team_sr': 'Team Strike Rate'}
    )
    fig_score_sr.update_layout(
        title='Average Score vs Strike Rate',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    
    charts_row1 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_total_runs, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_score_sr, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Top 3 Dependency - Sunburst
    team_batting_breakdown = team_stats[['team', 'top3_runs', 'total_runs']].copy()
    team_batting_breakdown['rest_runs'] = team_batting_breakdown['total_runs'] - team_batting_breakdown['top3_runs']
    
    # Prepare data for sunburst
    sunburst_data = []
    for _, row in team_batting_breakdown.iterrows():
        sunburst_data.append({'team': row['team'], 'category': 'Top 3', 'runs': row['top3_runs']})
        sunburst_data.append({'team': row['team'], 'category': 'Rest', 'runs': row['rest_runs']})
    
    sunburst_df = pd.DataFrame(sunburst_data)
    
    fig_sunburst = px.sunburst(
        sunburst_df,
        path=['team', 'category'],
        values='runs',
        color='category',
        color_discrete_map={'Top 3': COLORS['primary'], 'Rest': COLORS['tertiary']}
    )
    fig_sunburst.update_layout(
        title='Top 3 vs Rest Contribution by Team',
        font={'family': 'Arial'},
        height=500
    )
    
    # Top 3 Dependency Bar
    top3_dep = team_stats.nlargest(15, 'top3_dependency')[['team', 'top3_dependency']]
    
    fig_top3_dep = px.bar(
        top3_dep,
        x='team',
        y='top3_dependency',
        color='top3_dependency',
        color_continuous_scale='Reds',
        text=top3_dep['top3_dependency'].apply(lambda x: f'{x:.1f}%'),
        labels={'top3_dependency': 'Top 3 Dependency %', 'team': 'Team'}
    )
    fig_top3_dep.update_layout(
        title='Top 3 Batters Dependency Index',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_top3_dep.update_traces(textposition='outside')
    
    charts_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_sunburst, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_top3_dep, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # 50s and 100s by Team
    team_milestones = team_stats[['team', 'fifties', 'hundreds']].nlargest(15, 'hundreds')
    
    fig_milestones = go.Figure()
    fig_milestones.add_trace(go.Bar(
        name='Centuries',
        x=team_milestones['team'],
        y=team_milestones['hundreds'],
        marker_color=COLORS['gold'],
        text=team_milestones['hundreds'],
        textposition='outside'
    ))
    fig_milestones.add_trace(go.Bar(
        name='Fifties',
        x=team_milestones['team'],
        y=team_milestones['fifties'],
        marker_color=COLORS['tertiary'],
        text=team_milestones['fifties'],
        textposition='outside'
    ))
    fig_milestones.update_layout(
        title='Centuries and Fifties by Team',
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    
    # Boundary % Comparison
    boundary_data = team_stats.nlargest(15, 'boundary_pct')[['team', 'boundary_pct']]
    
    fig_boundary = px.bar(
        boundary_data,
        x='team',
        y='boundary_pct',
        color='boundary_pct',
        color_continuous_scale='Oranges',
        text=boundary_data['boundary_pct'].apply(lambda x: f'{x:.1f}%'),
        labels={'boundary_pct': 'Boundary %', 'team': 'Team'}
    )
    fig_boundary.update_layout(
        title='Boundary Percentage by Team',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_boundary.update_traces(textposition='outside')
    
    charts_row3 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_milestones, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_boundary, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Top Run Scorers Table
    top_scorers = player_batting.groupby('Batter').agg({
        'Runs': 'sum',
        'Balls': 'sum',
        '4s': 'sum',
        '6s': 'sum',
        'match_id': 'count'
    }).reset_index()
    top_scorers.columns = ['Batter', 'Runs', 'Balls', '4s', '6s', 'Innings']
    top_scorers['SR'] = (top_scorers['Runs'] / top_scorers['Balls'] * 100).round(2)
    top_scorers['Avg'] = (top_scorers['Runs'] / top_scorers['Innings']).round(2)
    top_scorers = top_scorers.nlargest(15, 'Runs')
    
    scorers_table = dbc.Card([
        dbc.CardHeader('Top 15 Run Scorers', style={'backgroundColor': COLORS['primary'], 'color': 'white', 'fontWeight': 'bold'}),
        dbc.CardBody([
            dash_table.DataTable(
                data=top_scorers.to_dict('records'),
                columns=[
                    {'name': 'Batter', 'id': 'Batter'},
                    {'name': 'Runs', 'id': 'Runs'},
                    {'name': 'Innings', 'id': 'Innings'},
                    {'name': 'Avg', 'id': 'Avg'},
                    {'name': 'SR', 'id': 'SR'},
                    {'name': '4s', 'id': '4s'},
                    {'name': '6s', 'id': '6s'},
                ],
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial'},
                style_header={'backgroundColor': COLORS['secondary'], 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
                ],
                page_size=15
            )
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    return html.Div([
        html.H2('🏏 Batting Analysis', className='mb-4', style={'color': COLORS['secondary']}),
        kpi_row,
        create_section_header('Team Batting Performance', 'Total runs and scoring patterns'),
        charts_row1,
        create_section_header('Batting Dependency Analysis', 'Top order contribution and milestones'),
        charts_row2,
        charts_row3,
        create_section_header('Top Performers', 'Individual batting leaders'),
        scorers_table
    ])


# =============================================================================
# PAGE 4: BOWLING ANALYSIS
# =============================================================================

def create_page_bowling():
    """Create Bowling Analysis page"""
    
    # Top KPIs
    avg_economy = team_stats['economy'].mean()
    avg_bowling_sr = team_stats['bowling_sr'].mean()
    avg_wickets_per_match = team_stats['wickets_per_match'].mean()
    
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card('Total Wickets', format_number(total_wickets), 
                          f'Avg: {avg_wickets_per_match:.1f}/match', COLORS['success'])
        ], md=3),
        dbc.Col([
            create_kpi_card('5-Wicket Hauls', str(total_5wkt_hauls), 
                          'Individual performances', COLORS['primary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Avg Economy', f'{avg_economy:.2f}', 
                          'Runs per over', COLORS['warning'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Avg Strike Rate', f'{avg_bowling_sr:.1f}', 
                          'Balls per wicket', COLORS['secondary'])
        ], md=3),
    ], className='mb-4')
    
    # Economy vs Wickets Bubble Chart
    fig_econ_wkts = px.scatter(
        team_stats,
        x='economy',
        y='total_wickets',
        size='matches_played',
        hover_name='team_name',
        hover_data=['bowling_avg', 'bowling_sr', 'five_wkt_hauls'],
        
        labels={'economy': 'Economy Rate', 'total_wickets': 'Total Wickets'}
    )
    fig_econ_wkts.update_layout(
        title='Economy Rate vs Total Wickets',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=450
    )
    
    # Best Economy Teams
    best_economy = team_stats.nsmallest(15, 'economy')[['team', 'economy']]
    
    fig_best_econ = px.bar(
        best_economy,
        x='economy',
        y='team',
        orientation='h',
        color='economy',
        color_continuous_scale='Greens_r',
        text=best_economy['economy'].apply(lambda x: f'{x:.2f}'),
        labels={'economy': 'Economy Rate', 'team': 'Team'}
    )
    fig_best_econ.update_layout(
        title='Best Economy Rates (Lower is Better)',
        yaxis={'categoryorder': 'total descending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=450
    )
    fig_best_econ.update_traces(textposition='outside')
    
    charts_row1 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_econ_wkts, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_best_econ, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Bowling Average vs Strike Rate
    fig_avg_sr = px.scatter(
        team_stats[team_stats['bowling_avg'] < 100],  # Filter outliers
        x='bowling_avg',
        y='bowling_sr',
        size='total_wickets',
        hover_name='team_name',
        hover_data=['economy', 'wickets_per_match'],
        
        labels={'bowling_avg': 'Bowling Average', 'bowling_sr': 'Bowling Strike Rate'}
    )
    fig_avg_sr.update_layout(
        title='Bowling Average vs Strike Rate (Lower is Better for Both)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    
    # Maidens per Match
    maidens_data = team_stats.nlargest(15, 'maidens_per_match')[['team', 'maidens_per_match']]
    
    fig_maidens = px.bar(
        maidens_data,
        x='team',
        y='maidens_per_match',
        color='maidens_per_match',
        color_continuous_scale='Blues',
        text=maidens_data['maidens_per_match'].apply(lambda x: f'{x:.2f}'),
        labels={'maidens_per_match': 'Maidens/Match', 'team': 'Team'}
    )
    fig_maidens.update_layout(
        title='Maidens per Match (Dot Ball Proxy)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_maidens.update_traces(textposition='outside')
    
    charts_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_avg_sr, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_maidens, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # 5-Wicket Hauls by Team
    five_wkt_data = team_stats[team_stats['five_wkt_hauls'] > 0].sort_values('five_wkt_hauls', ascending=False)[['team', 'five_wkt_hauls']]
    
    fig_5wkt = px.bar(
        five_wkt_data,
        x='team',
        y='five_wkt_hauls',
        color='five_wkt_hauls',
        color_continuous_scale='Reds',
        text='five_wkt_hauls',
        labels={'five_wkt_hauls': '5-Wicket Hauls', 'team': 'Team'}
    )
    fig_5wkt.update_layout(
        title='5-Wicket Hauls by Team',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_5wkt.update_traces(textposition='outside')
    
    # Death Bowling Economy
    death_data = team_stats.nsmallest(15, 'death_bowling_economy')[['team', 'death_bowling_economy']]
    
    fig_death = px.bar(
        death_data,
        x='death_bowling_economy',
        y='team',
        orientation='h',
        color='death_bowling_economy',
        color_continuous_scale='RdYlGn_r',
        text=death_data['death_bowling_economy'].apply(lambda x: f'{x:.2f}'),
        labels={'death_bowling_economy': 'Death Bowling Economy', 'team': 'Team'}
    )
    fig_death.update_layout(
        title='Death Bowling Economy (Last 3 Bowlers Avg)',
        yaxis={'categoryorder': 'total descending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_death.update_traces(textposition='outside')
    
    charts_row3 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_5wkt, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_death, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Top Wicket Takers Table
    top_bowlers = player_bowling.groupby('Bowler').agg({
        'Wickets': 'sum',
        'Runs': 'sum',
        'Overs': 'sum',
        'Maidens': 'sum',
        'match_id': 'count'
    }).reset_index()
    top_bowlers.columns = ['Bowler', 'Wickets', 'Runs', 'Overs', 'Maidens', 'Matches']
    top_bowlers['Economy'] = (top_bowlers['Runs'] / top_bowlers['Overs']).round(2)
    top_bowlers['Avg'] = (top_bowlers['Runs'] / top_bowlers['Wickets']).round(2)
    top_bowlers = top_bowlers[top_bowlers['Wickets'] > 0].nlargest(15, 'Wickets')
    
    bowlers_table = dbc.Card([
        dbc.CardHeader('Top 15 Wicket Takers', style={'backgroundColor': COLORS['success'], 'color': 'white', 'fontWeight': 'bold'}),
        dbc.CardBody([
            dash_table.DataTable(
                data=top_bowlers.to_dict('records'),
                columns=[
                    {'name': 'Bowler', 'id': 'Bowler'},
                    {'name': 'Wickets', 'id': 'Wickets'},
                    {'name': 'Matches', 'id': 'Matches'},
                    {'name': 'Overs', 'id': 'Overs'},
                    {'name': 'Runs', 'id': 'Runs'},
                    {'name': 'Avg', 'id': 'Avg'},
                    {'name': 'Economy', 'id': 'Economy'},
                ],
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial'},
                style_header={'backgroundColor': COLORS['secondary'], 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
                ],
                page_size=15
            )
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    return html.Div([
        html.H2('🎯 Bowling Analysis', className='mb-4', style={'color': COLORS['secondary']}),
        kpi_row,
        create_section_header('Team Bowling Performance', 'Economy and wicket-taking ability'),
        charts_row1,
        create_section_header('Bowling Efficiency Metrics', 'Average, strike rate, and maidens'),
        charts_row2,
        create_section_header('Special Performances', '5-wicket hauls and death bowling'),
        charts_row3,
        create_section_header('Top Performers', 'Individual bowling leaders'),
        bowlers_table
    ])


# =============================================================================
# PAGE 5: TOSS & VENUE IMPACT
# =============================================================================

def create_page_toss_venue():
    """Create Toss & Venue Impact page"""
    
    # Toss Statistics
    toss_bat = len(matches[matches['toss_decision'] == 'Bat'])
    toss_bowl = len(matches[matches['toss_decision'] == 'Bowl'])
    
    # Win after winning toss
    toss_win_match_win = len(matches[matches['toss_winner'] == matches['winner']])
    toss_win_match_win_pct = toss_win_match_win / total_matches * 100
    
    # Bat first wins vs Bowl first wins
    bat_first_wins = len(matches[matches['batting_first'] == matches['winner']])
    bowl_first_wins = total_matches - bat_first_wins
    
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card('Chose to Bat', str(toss_bat), 
                          f'{toss_bat/total_matches*100:.1f}% of tosses', COLORS['primary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Chose to Bowl', str(toss_bowl), 
                          f'{toss_bowl/total_matches*100:.1f}% of tosses', COLORS['secondary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Toss Winner Won Match', f'{toss_win_match_win_pct:.1f}%', 
                          f'{toss_win_match_win} of {total_matches} matches', COLORS['success'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Bat First Wins', str(bat_first_wins), 
                          f'{bat_first_wins/total_matches*100:.1f}% | Bowl First: {bowl_first_wins}', COLORS['warning'])
        ], md=3),
    ], className='mb-4')
    
    # Sankey Diagram: Toss Winner -> Decision -> Match Result
    # Prepare data for Sankey
    sankey_data = matches.copy()
    sankey_data['toss_result'] = sankey_data.apply(
        lambda x: 'Won Match' if x['toss_winner'] == x['winner'] else 'Lost Match', axis=1
    )
    
    # Count combinations
    sankey_counts = sankey_data.groupby(['toss_decision', 'toss_result']).size().reset_index(name='count')
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=['Won Toss', 'Bat First', 'Bowl First', 'Won Match', 'Lost Match'],
            color=[COLORS['gold'], COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2],
            target=[1, 2, 3, 4, 3, 4],
            value=[
                toss_bat, toss_bowl,
                len(sankey_data[(sankey_data['toss_decision']=='Bat') & (sankey_data['toss_result']=='Won Match')]),
                len(sankey_data[(sankey_data['toss_decision']=='Bat') & (sankey_data['toss_result']=='Lost Match')]),
                len(sankey_data[(sankey_data['toss_decision']=='Bowl') & (sankey_data['toss_result']=='Won Match')]),
                len(sankey_data[(sankey_data['toss_decision']=='Bowl') & (sankey_data['toss_result']=='Lost Match')])
            ],
            color=['rgba(230,57,70,0.4)', 'rgba(29,53,87,0.4)', 
                   'rgba(42,157,143,0.6)', 'rgba(244,162,97,0.6)',
                   'rgba(42,157,143,0.6)', 'rgba(244,162,97,0.6)']
        )
    )])
    fig_sankey.update_layout(
        title='Toss Decision Flow: Decision → Match Result',
        font={'family': 'Arial'},
        height=400
    )
    
    # Bat First vs Bowl First Win %
    fig_bat_bowl = go.Figure()
    fig_bat_bowl.add_trace(go.Pie(
        labels=['Bat First Wins', 'Bowl First Wins'],
        values=[bat_first_wins, bowl_first_wins],
        hole=0.5,
        marker_colors=[COLORS['primary'], COLORS['secondary']],
        textinfo='label+percent',
        textposition='outside'
    ))
    fig_bat_bowl.update_layout(
        title='Batting First vs Bowling First Wins',
        font={'family': 'Arial'},
        height=400,
        showlegend=False
    )
    
    charts_row1 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_sankey, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_bat_bowl, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=5),
    ], className='mb-4')
    
    # Venue-wise Average First Innings Score
    venue_avg = venue_stats.sort_values('avg_first_innings', ascending=False).head(15)
    
    fig_venue_avg = px.bar(
        venue_avg,
        x='avg_first_innings',
        y='venue',
        orientation='h',
        color='avg_first_innings',
        color_continuous_scale='Reds',
        text=venue_avg['avg_first_innings'].apply(lambda x: f'{x:.0f}'),
        labels={'avg_first_innings': 'Avg 1st Innings Score', 'venue': 'Venue'}
    )
    fig_venue_avg.update_layout(
        title='Average First Innings Score by Venue',
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=500
    )
    fig_venue_avg.update_traces(textposition='outside')
    
    # Chase vs Defend Success by Venue
    venue_chase = venue_stats.sort_values('chase_success_rate', ascending=False).head(15)
    
    fig_chase_defend = go.Figure()
    fig_chase_defend.add_trace(go.Bar(
        name='Chase Success %',
        y=venue_chase['venue'],
        x=venue_chase['chase_success_rate'],
        orientation='h',
        marker_color=COLORS['success'],
        text=venue_chase['chase_success_rate'].apply(lambda x: f'{x:.0f}%'),
        textposition='outside'
    ))
    fig_chase_defend.add_trace(go.Bar(
        name='Defend Success %',
        y=venue_chase['venue'],
        x=venue_chase['defend_success_rate'],
        orientation='h',
        marker_color=COLORS['primary'],
        text=venue_chase['defend_success_rate'].apply(lambda x: f'{x:.0f}%'),
        textposition='outside'
    ))
    fig_chase_defend.update_layout(
        title='Chase vs Defend Success Rate by Venue',
        barmode='group',
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=500
    )
    
    charts_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_venue_avg, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_chase_defend, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Venue Statistics Table
    venue_table_data = venue_stats[['venue', 'total_matches', 'avg_first_innings', 'chase_success_rate', 'defend_success_rate', 'high_scoring_matches']].copy()
    venue_table_data.columns = ['Venue', 'Matches', 'Avg 1st Inn', 'Chase %', 'Defend %', 'High Scoring']
    venue_table_data = venue_table_data.sort_values('Matches', ascending=False)
    
    venue_table = dbc.Card([
        dbc.CardHeader('Venue Statistics Summary', style={'backgroundColor': COLORS['secondary'], 'color': 'white', 'fontWeight': 'bold'}),
        dbc.CardBody([
            dash_table.DataTable(
                data=venue_table_data.to_dict('records'),
                columns=[
                    {'name': 'Venue', 'id': 'Venue'},
                    {'name': 'Matches', 'id': 'Matches'},
                    {'name': 'Avg 1st Inn', 'id': 'Avg 1st Inn', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                    {'name': 'Chase %', 'id': 'Chase %', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Defend %', 'id': 'Defend %', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'High Scoring', 'id': 'High Scoring'},
                ],
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial', 'whiteSpace': 'normal', 'height': 'auto'},
                style_header={'backgroundColor': COLORS['tertiary'], 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
                ],
                page_size=15
            )
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    return html.Div([
        html.H2('🎲 Toss & Venue Impact', className='mb-4', style={'color': COLORS['secondary']}),
        kpi_row,
        create_section_header('Toss Analysis', 'Decision patterns and outcomes'),
        charts_row1,
        create_section_header('Venue Analysis', 'Scoring patterns and chase/defend success'),
        charts_row2,
        venue_table
    ])


# =============================================================================
# PAGE 6: MATCH SITUATIONS
# =============================================================================

def create_page_match_situations():
    """Create Match Situations page"""
    
    # Win margin analysis
    runs_wins = matches[matches['win_type'] == 'runs'].copy()
    wickets_wins = matches[matches['win_type'] == 'wickets'].copy()
    
    # Pressure wins (< 20 runs or < 3 wickets)
    pressure_runs = len(runs_wins[runs_wins['win_margin'] < 20])
    pressure_wkts = len(wickets_wins[wickets_wins['win_margin'] < 3])
    pressure_total = pressure_runs + pressure_wkts
    
    # Dominant wins (80+ runs or 7+ wickets)
    dominant_runs = len(runs_wins[runs_wins['win_margin'] >= 80])
    dominant_wkts = len(wickets_wins[wickets_wins['win_margin'] >= 7])
    dominant_total = dominant_runs + dominant_wkts
    
    # Comfortable wins (in between)
    comfortable_total = total_matches - pressure_total - dominant_total
    
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card('Pressure Wins', str(pressure_total), 
                          '<20 runs or <3 wickets', COLORS['primary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Comfortable Wins', str(comfortable_total), 
                          'Middle ground', COLORS['tertiary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Dominant Wins', str(dominant_total), 
                          '80+ runs or 7+ wickets', COLORS['success'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Avg Win Margin', f"{runs_wins['win_margin'].mean():.0f} runs", 
                          f"or {wickets_wins['win_margin'].mean():.1f} wickets", COLORS['warning'])
        ], md=3),
    ], className='mb-4')
    
    # Win Margin Distribution - Runs
    fig_margin_runs = px.histogram(
        runs_wins,
        x='win_margin',
        nbins=20,
        color_discrete_sequence=[COLORS['primary']],
        labels={'win_margin': 'Win Margin (Runs)', 'count': 'Matches'}
    )
    fig_margin_runs.update_layout(
        title='Win Margin Distribution (By Runs)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=350
    )
    
    # Win Margin Distribution - Wickets
    fig_margin_wkts = px.histogram(
        wickets_wins,
        x='win_margin',
        nbins=10,
        color_discrete_sequence=[COLORS['secondary']],
        labels={'win_margin': 'Win Margin (Wickets)', 'count': 'Matches'}
    )
    fig_margin_wkts.update_layout(
        title='Win Margin Distribution (By Wickets)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=350
    )
    
    charts_row1 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_margin_runs, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_margin_wkts, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Win Type Breakdown
    fig_win_type = go.Figure()
    fig_win_type.add_trace(go.Pie(
        labels=['Pressure Wins', 'Comfortable Wins', 'Dominant Wins'],
        values=[pressure_total, comfortable_total, dominant_total],
        hole=0.5,
        marker_colors=[COLORS['warning'], COLORS['tertiary'], COLORS['success']],
        textinfo='label+percent+value',
        textposition='outside'
    ))
    fig_win_type.update_layout(
        title='Match Results by Win Type',
        font={'family': 'Arial'},
        height=400,
        showlegend=False
    )
    
    # Teams with Most Pressure Wins
    pressure_wins_team = team_stats.nlargest(10, 'pressure_wins')[['team', 'pressure_wins', 'dominant_wins']]
    
    fig_pressure_teams = go.Figure()
    fig_pressure_teams.add_trace(go.Bar(
        name='Pressure Wins',
        x=pressure_wins_team['team'],
        y=pressure_wins_team['pressure_wins'],
        marker_color=COLORS['warning'],
        text=pressure_wins_team['pressure_wins'],
        textposition='outside'
    ))
    fig_pressure_teams.add_trace(go.Bar(
        name='Dominant Wins',
        x=pressure_wins_team['team'],
        y=pressure_wins_team['dominant_wins'],
        marker_color=COLORS['success'],
        text=pressure_wins_team['dominant_wins'],
        textposition='outside'
    ))
    fig_pressure_teams.update_layout(
        title='Pressure vs Dominant Wins by Team',
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    
    charts_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_win_type, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_pressure_teams, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=7),
    ], className='mb-4')
    
    # Closest Finishes
    closest_runs = runs_wins.nsmallest(5, 'win_margin')[['match_id', 'team1', 'team2', 'team1_runs', 'team2_runs', 'winner', 'win_margin', 'stage']]
    closest_wkts = wickets_wins.nsmallest(5, 'win_margin')[['match_id', 'team1', 'team2', 'team1_runs', 'team2_runs', 'winner', 'win_margin', 'stage']]
    
    # Biggest Victories
    biggest_runs = runs_wins.nlargest(5, 'win_margin')[['match_id', 'team1', 'team2', 'team1_runs', 'team2_runs', 'winner', 'win_margin', 'stage']]
    biggest_wkts = wickets_wins.nlargest(5, 'win_margin')[['match_id', 'team1', 'team2', 'team1_runs', 'team2_runs', 'winner', 'win_margin', 'stage']]
    
    # Create cards for closest finishes
    closest_cards = []
    for _, row in closest_runs.head(3).iterrows():
        closest_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{row['team1']} vs {row['team2']}", className='mb-1'),
                        html.P(f"{int(row['team1_runs'])} vs {int(row['team2_runs'])}", className='text-muted mb-1'),
                        html.P(f"{row['winner']} won by {int(row['win_margin'])} runs", className='fw-bold', style={'color': COLORS['primary']}),
                        html.Small(row['stage'], className='text-muted')
                    ])
                ], className='shadow-sm h-100', style={'borderRadius': '10px', 'border': f'2px solid {COLORS["warning"]}'})
            ], md=4)
        )
    
    closest_section = html.Div([
        create_section_header('Closest Finishes (By Runs)', 'Nail-biting encounters'),
        dbc.Row(closest_cards, className='mb-4')
    ])
    
    # Biggest victories cards
    biggest_cards = []
    for _, row in biggest_runs.head(3).iterrows():
        biggest_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{row['team1']} vs {row['team2']}", className='mb-1'),
                        html.P(f"{int(row['team1_runs'])} vs {int(row['team2_runs'])}", className='text-muted mb-1'),
                        html.P(f"{row['winner']} won by {int(row['win_margin'])} runs", className='fw-bold', style={'color': COLORS['success']}),
                        html.Small(row['stage'], className='text-muted')
                    ])
                ], className='shadow-sm h-100', style={'borderRadius': '10px', 'border': f'2px solid {COLORS["success"]}'})
            ], md=4)
        )
    
    biggest_section = html.Div([
        create_section_header('Biggest Victories (By Runs)', 'One-sided encounters'),
        dbc.Row(biggest_cards, className='mb-4')
    ])
    
    # Chase Analysis - Target vs Result
    chase_data = matches.copy()
    chase_data['target'] = chase_data['team1_runs'] + 1
    chase_data['chase_success'] = chase_data['winner'] != chase_data['batting_first']
    
    fig_chase = px.scatter(
        chase_data,
        x='target',
        y='team2_runs',
        hover_data=['team1', 'team2', 'winner'],
        labels={'target': 'Target', 'team2_runs': 'Chasing Team Score', 'chase_success': 'Chase Successful'}
    )
    fig_chase.add_shape(type='line', x0=0, y0=0, x1=400, y1=400, line=dict(color='gray', dash='dash'))
    fig_chase.update_layout(
        title='Chase Analysis: Target vs Chasing Team Score',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=450
    )
    
    chase_card = dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig_chase, config={'displayModeBar': False})
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    return html.Div([
        html.H2('⚔️ Match Situations', className='mb-4', style={'color': COLORS['secondary']}),
        kpi_row,
        create_section_header('Win Margin Distribution', 'How matches were won'),
        charts_row1,
        charts_row2,
        closest_section,
        biggest_section,
        create_section_header('Chase Analysis', 'Target vs actual score'),
        chase_card
    ])


# =============================================================================
# PAGE 7: DISMISSAL PATTERNS
# =============================================================================

def create_page_dismissals():
    """Create Dismissal Patterns page"""
    
    # Calculate dismissal statistics
    dismissal_counts = player_batting['dismissal_type'].value_counts()
    total_dismissals = dismissal_counts.drop(['not out', 'batting'], errors='ignore').sum()
    
    caught_count = dismissal_counts.get('caught', 0)
    bowled_count = dismissal_counts.get('bowled', 0)
    lbw_count = dismissal_counts.get('lbw', 0)
    stumped_count = dismissal_counts.get('stumped', 0)
    run_out_count = dismissal_counts.get('run out', 0)
    
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card('Caught', str(caught_count), 
                          f'{caught_count/total_dismissals*100:.1f}%', COLORS['primary'])
        ], md=2),
        dbc.Col([
            create_kpi_card('Bowled', str(bowled_count), 
                          f'{bowled_count/total_dismissals*100:.1f}%', COLORS['secondary'])
        ], md=2),
        dbc.Col([
            create_kpi_card('LBW', str(lbw_count), 
                          f'{lbw_count/total_dismissals*100:.1f}%', COLORS['tertiary'])
        ], md=2),
        dbc.Col([
            create_kpi_card('Stumped', str(stumped_count), 
                          f'{stumped_count/total_dismissals*100:.1f}%', COLORS['success'])
        ], md=2),
        dbc.Col([
            create_kpi_card('Run Out', str(run_out_count), 
                          f'{run_out_count/total_dismissals*100:.1f}%', COLORS['warning'])
        ], md=2),
        dbc.Col([
            create_kpi_card('Total', str(total_dismissals), 
                          'All dismissals', COLORS['purple'])
        ], md=2),
    ], className='mb-4')
    
    # Overall Dismissal Pie Chart
    dismissal_for_pie = dismissal_counts.drop(['not out', 'batting', 'unknown', 'retired hurt', 'hit wicket', 'absent hurt'], errors='ignore')
    
    fig_dismissal_pie = px.pie(
        values=dismissal_for_pie.values,
        names=dismissal_for_pie.index,
        color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
                                  COLORS['success'], COLORS['warning']]
    )
    fig_dismissal_pie.update_layout(
        title='Overall Dismissal Type Distribution',
        font={'family': 'Arial'},
        height=400
    )
    
    # Dismissal by Team (How they got batters out - bowling effectiveness)
    team_dismissals = player_batting.groupby(['Team', 'dismissal_type']).size().unstack(fill_value=0)
    team_dismissals = team_dismissals.drop(['not out', 'batting', 'unknown'], axis=1, errors='ignore')
    team_dismissals['total'] = team_dismissals.sum(axis=1)
    team_dismissals = team_dismissals.nlargest(15, 'total').drop('total', axis=1)
    
    fig_team_dismissals = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['success'], COLORS['warning'], COLORS['purple']]
    
    for i, col in enumerate(team_dismissals.columns):
        fig_team_dismissals.add_trace(go.Bar(
            name=col.title(),
            x=team_dismissals.index,
            y=team_dismissals[col],
            marker_color=colors[i % len(colors)]
        ))
    
    fig_team_dismissals.update_layout(
        title='Dismissal Types by Team (How They Got Out)',
        barmode='stack',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=450
    )
    
    charts_row1 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_dismissal_pie, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_team_dismissals, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=7),
    ], className='mb-4')
    
    # Caught % by Team (Fielding Effectiveness)
    caught_by_team = team_stats.nlargest(15, 'caught_pct')[['team', 'caught_pct']]
    
    fig_caught = px.bar(
        caught_by_team,
        x='team',
        y='caught_pct',
        color='caught_pct',
        color_continuous_scale='Blues',
        text=caught_by_team['caught_pct'].apply(lambda x: f'{x:.1f}%'),
        labels={'caught_pct': 'Caught %', 'team': 'Team'}
    )
    fig_caught.update_layout(
        title='Caught Dismissal % by Team (Higher = More Catches Taken)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_caught.update_traces(textposition='outside')
    
    # Bowled % by Team
    bowled_by_team = team_stats.nlargest(15, 'bowled_pct')[['team', 'bowled_pct']]
    
    fig_bowled = px.bar(
        bowled_by_team,
        x='team',
        y='bowled_pct',
        color='bowled_pct',
        color_continuous_scale='Greens',
        text=bowled_by_team['bowled_pct'].apply(lambda x: f'{x:.1f}%'),
        labels={'bowled_pct': 'Bowled %', 'team': 'Team'}
    )
    fig_bowled.update_layout(
        title='Bowled Dismissal % by Team (Higher = Hit Stumps More)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_bowled.update_traces(textposition='outside')
    
    charts_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_caught, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_bowled, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # LBW % by Team
    lbw_by_team = team_stats.nlargest(15, 'lbw_pct')[['team', 'lbw_pct']]
    
    fig_lbw = px.bar(
        lbw_by_team,
        x='team',
        y='lbw_pct',
        color='lbw_pct',
        color_continuous_scale='Reds',
        text=lbw_by_team['lbw_pct'].apply(lambda x: f'{x:.1f}%'),
        labels={'lbw_pct': 'LBW %', 'team': 'Team'}
    )
    fig_lbw.update_layout(
        title='LBW Dismissal % by Team',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_lbw.update_traces(textposition='outside')
    
    lbw_card = dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig_lbw, config={'displayModeBar': False})
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    return html.Div([
        html.H2('🧤 Dismissal Patterns', className='mb-4', style={'color': COLORS['secondary']}),
        kpi_row,
        create_section_header('Dismissal Overview', 'How wickets fell across the tournament'),
        charts_row1,
        create_section_header('Team-wise Dismissal Analysis', 'Fielding and bowling effectiveness'),
        charts_row2,
        lbw_card
    ])


# =============================================================================
# PAGE 8: QUALIFIED TEAMS DEEP DIVE
# =============================================================================

def create_page_qualified():
    """Create Qualified Teams Deep Dive page"""
    
    # Get qualified teams
    qualified_teams = team_stats[team_stats['qualified'] == True].copy()
    eliminated_teams = team_stats[team_stats['qualified'] == False].copy()
    
    # KPIs - Qualified vs Eliminated comparison
    qual_avg_score = qualified_teams['avg_score'].mean()
    elim_avg_score = eliminated_teams['avg_score'].mean()
    qual_economy = qualified_teams['economy'].mean()
    elim_economy = eliminated_teams['economy'].mean()
    qual_win_pct = qualified_teams['win_pct'].mean()
    elim_win_pct = eliminated_teams['win_pct'].mean()
    
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card('Qualified Teams', '10', 
                          '8 Main + 2 Plate', COLORS['gold'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Avg Score (Q vs E)', f'{qual_avg_score:.0f} vs {elim_avg_score:.0f}', 
                          'Qualified higher', COLORS['success'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Avg Economy (Q vs E)', f'{qual_economy:.2f} vs {elim_economy:.2f}', 
                          'Qualified lower (better)', COLORS['primary'])
        ], md=3),
        dbc.Col([
            create_kpi_card('Avg Win % (Q vs E)', f'{qual_win_pct:.0f}% vs {elim_win_pct:.0f}%', 
                          'Clear difference', COLORS['secondary'])
        ], md=3),
    ], className='mb-4')
    
    # Qualified Teams Cards
    qualified_list = qualified_teams[['team', 'team_name', 'group', 'wins', 'win_pct', 'nrr']].sort_values('win_pct', ascending=False)
    
    qualified_cards = []
    for _, row in qualified_list.iterrows():
        color = COLORS['gold'] if row['team'] == 'VID' else COLORS['tertiary']
        badge = '🏆' if row['team'] == 'VID' else ('🥈' if row['team'] == 'SAUR' else '')
        qualified_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{badge} {row['team']}", className='mb-1', style={'color': color}),
                        html.P(row['team_name'], className='text-muted mb-1', style={'fontSize': '0.85rem'}),
                        html.P(f"{row['group']}", className='mb-1'),
                        html.P(f"Win%: {row['win_pct']:.1f}% | NRR: {row['nrr']:+.3f}", className='fw-bold mb-0')
                    ])
                ], className='shadow-sm h-100', style={'borderRadius': '10px', 'border': f'2px solid {color}'})
            ], md=2, className='mb-3')
        )
    
    qualified_row = dbc.Row(qualified_cards, className='mb-4')
    
    # Radar Chart for Qualified Teams comparison
    # Normalize metrics for radar
    metrics = ['win_pct', 'avg_score', 'team_sr', 'economy', 'wickets_per_match', 'nrr']
    
    # Create radar chart for top 4 qualified teams
    top_qualified = qualified_teams.nlargest(4, 'win_pct')
    
    fig_radar = go.Figure()
    
    for _, row in top_qualified.iterrows():
        values = [
            row['win_pct'] / 100,
            row['avg_score'] / 350,
            row['team_sr'] / 120,
            1 - (row['economy'] / 10),  # Invert economy
            row['wickets_per_match'] / 10,
            (row['nrr'] + 3) / 6  # Normalize NRR
        ]
        values.append(values[0])  # Close the polygon
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=['Win %', 'Avg Score', 'Strike Rate', 'Economy', 'Wickets/Match', 'NRR', 'Win %'],
            fill='toself',
            name=row['team'],
            line_color=get_team_color(row['team'])
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Top 4 Qualified Teams Comparison',
        font={'family': 'Arial'},
        height=500,
        showlegend=True
    )
    
    # Qualified vs Eliminated comparison bar
    comparison_data = pd.DataFrame({
        'Metric': ['Avg Score', 'Strike Rate', 'Economy', 'Wickets/Match', 'Win %'],
        'Qualified': [qual_avg_score, qualified_teams['team_sr'].mean(), qual_economy, 
                      qualified_teams['wickets_per_match'].mean(), qual_win_pct],
        'Eliminated': [elim_avg_score, eliminated_teams['team_sr'].mean(), elim_economy,
                       eliminated_teams['wickets_per_match'].mean(), elim_win_pct]
    })
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        name='Qualified',
        x=comparison_data['Metric'],
        y=comparison_data['Qualified'],
        marker_color=COLORS['success'],
        text=comparison_data['Qualified'].round(1),
        textposition='outside'
    ))
    fig_comparison.add_trace(go.Bar(
        name='Eliminated',
        x=comparison_data['Metric'],
        y=comparison_data['Eliminated'],
        marker_color=COLORS['warning'],
        text=comparison_data['Eliminated'].round(1),
        textposition='outside'
    ))
    fig_comparison.update_layout(
        title='Qualified vs Eliminated Teams - Key Metrics',
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    
    charts_row1 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_radar, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_comparison, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Balanced Team Index
    qualified_balanced = qualified_teams.sort_values('balanced_index')[['team', 'team_name', 'batting_rank', 'bowling_rank', 'balanced_index']]
    
    fig_balanced = px.bar(
        qualified_balanced,
        x='team',
        y='balanced_index',
        color='balanced_index',
        color_continuous_scale='RdYlGn_r',
        text=qualified_balanced['balanced_index'].apply(lambda x: f'{x:.1f}'),
        labels={'balanced_index': 'Balanced Index (Lower = Better)', 'team': 'Team'}
    )
    fig_balanced.update_layout(
        title='Balanced Team Index (Batting Rank + Bowling Rank)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    fig_balanced.update_traces(textposition='outside')
    
    balanced_card = dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig_balanced, config={'displayModeBar': False})
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    # Qualification Margin Table
    qual_margin_data = []
    for group in standings['Group'].unique():
        group_standings = standings[standings['Group'] == group].sort_values('Pos')
        if len(group_standings) >= 3:
            second_place = group_standings.iloc[1]
            third_place = group_standings.iloc[2]
            margin = second_place['Pts'] - third_place['Pts']
            qual_margin_data.append({
                'Group': group,
                '2nd Place': second_place['Team_Clean'],
                '2nd Pts': second_place['Pts'],
                '3rd Place': third_place['Team_Clean'],
                '3rd Pts': third_place['Pts'],
                'Margin': margin
            })
    
    qual_margin_df = pd.DataFrame(qual_margin_data)
    
    margin_table = dbc.Card([
        dbc.CardHeader('Qualification Margin by Group', style={'backgroundColor': COLORS['secondary'], 'color': 'white', 'fontWeight': 'bold'}),
        dbc.CardBody([
            dash_table.DataTable(
                data=qual_margin_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in qual_margin_df.columns],
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial'},
                style_header={'backgroundColor': COLORS['tertiary'], 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
                ]
            )
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    return html.Div([
        html.H2('🎖️ Qualified Teams Deep Dive', className='mb-4', style={'color': COLORS['secondary']}),
        kpi_row,
        create_section_header('The 10 Qualified Teams', 'Teams that made it to knockouts'),
        qualified_row,
        create_section_header('Performance Comparison', 'Qualified vs Eliminated analysis'),
        charts_row1,
        create_section_header('Balance Analysis', 'Teams with balanced batting and bowling'),
        balanced_card,
        margin_table
    ])


# =============================================================================
# PAGE 9: CHAMPION'S JOURNEY (VIDARBHA)
# =============================================================================

def create_page_champions():
    """Create Champion's Journey page - Vidarbha"""
    
    # Get Vidarbha matches
    vid_matches = matches[(matches['team1'] == 'VID') | (matches['team2'] == 'VID')].copy()
    vid_matches = vid_matches.sort_values(['month', 'day_num'])
    
    # Vidarbha stats
    vid_stats = team_stats[team_stats['team'] == 'VID'].iloc[0]
    
    # Champion Header
    champion_header = dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span('🏆', style={'fontSize': '4rem'}),
                html.H2('VIDARBHA - VIJAY HAZARE TROPHY 2025-26 CHAMPIONS', 
                       className='mb-2 mt-2', style={'color': COLORS['gold'], 'fontWeight': 'bold'}),
                html.H5(f"Final: VID 317-8 (50 ov) beat SAUR 279 (48.5 ov) by 38 runs", className='text-muted'),
                html.P('First Vijay Hazare Trophy title for Vidarbha!', className='mb-0 mt-2')
            ], className='text-center')
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': f'3px solid {COLORS["gold"]}', 
                                           'backgroundColor': '#fffbeb'})
    
    # Stats KPIs
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card('Matches', str(int(vid_stats['matches_played'])), 
                          f"{int(vid_stats['wins'])} Wins, {int(vid_stats['losses'])} Losses", COLORS['primary'])
        ], md=2),
        dbc.Col([
            create_kpi_card('Win %', f"{vid_stats['win_pct']:.1f}%", 
                          'Tournament win rate', COLORS['success'])
        ], md=2),
        dbc.Col([
            create_kpi_card('Avg Score', f"{vid_stats['avg_score']:.0f}", 
                          'Runs per match', COLORS['gold'])
        ], md=2),
        dbc.Col([
            create_kpi_card('Total Runs', format_number(vid_stats['total_runs']), 
                          f"{int(vid_stats['hundreds'])} 100s, {int(vid_stats['fifties'])} 50s", COLORS['secondary'])
        ], md=2),
        dbc.Col([
            create_kpi_card('Economy', f"{vid_stats['economy']:.2f}", 
                          'Bowling economy', COLORS['tertiary'])
        ], md=2),
        dbc.Col([
            create_kpi_card('NRR', f"{vid_stats['nrr']:+.3f}", 
                          'Net Run Rate', COLORS['purple'])
        ], md=2),
    ], className='mb-4')
    
    # Match-by-Match Timeline
    timeline_data = []
    cumulative_wins = 0
    cumulative_points = 0
    
    for _, match in vid_matches.iterrows():
        is_winner = match['winner'] == 'VID'
        if is_winner:
            cumulative_wins += 1
        if match['stage'] == 'Group Stage':
            cumulative_points += 4 if is_winner else 0
        
        opponent = match['team2'] if match['team1'] == 'VID' else match['team1']
        vid_score = match['team1_runs'] if match['team1'] == 'VID' else match['team2_runs']
        opp_score = match['team2_runs'] if match['team1'] == 'VID' else match['team1_runs']
        
        result = 'Won' if is_winner else 'Lost'
        margin = f"by {int(match['win_margin'])} {match['win_type']}" if is_winner else f"by {int(match['win_margin'])} {match['win_type']}"
        
        timeline_data.append({
            'match_num': len(timeline_data) + 1,
            'date': f"{match['month']} {int(match['day_num'])}",
            'opponent': opponent,
            'vid_score': int(vid_score),
            'opp_score': int(opp_score),
            'result': result,
            'margin': margin,
            'stage': match['stage'],
            'cumulative_wins': cumulative_wins,
            'cumulative_points': cumulative_points
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    # Points Progression Chart
    fig_progression = go.Figure()
    fig_progression.add_trace(go.Scatter(
        x=timeline_df['match_num'],
        y=timeline_df['cumulative_wins'],
        mode='lines+markers',
        name='Cumulative Wins',
        line=dict(color=COLORS['success'], width=3),
        marker=dict(size=10)
    ))
    fig_progression.update_layout(
        title="Vidarbha's Win Progression Through Tournament",
        xaxis_title='Match Number',
        yaxis_title='Cumulative Wins',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=350
    )
    
    # Scores Timeline
    fig_scores = go.Figure()
    fig_scores.add_trace(go.Bar(
        name='Vidarbha Score',
        x=timeline_df['match_num'],
        y=timeline_df['vid_score'],
        marker_color=COLORS['primary'],
        text=timeline_df['vid_score'],
        textposition='outside'
    ))
    fig_scores.add_trace(go.Bar(
        name='Opponent Score',
        x=timeline_df['match_num'],
        y=timeline_df['opp_score'],
        marker_color=COLORS['tertiary'],
        text=timeline_df['opp_score'],
        textposition='outside'
    ))
    fig_scores.update_layout(
        title='Match-by-Match Scores',
        xaxis_title='Match Number',
        yaxis_title='Score',
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=350
    )
    
    charts_row1 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_progression, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_scores, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')
    
    # Knockout Journey Funnel
    knockout_stages = ['Quarter Final', 'Semi Final', 'Final', 'CHAMPION']
    knockout_values = [4, 2, 1, 1]  # Number of teams at each stage
    
    fig_funnel = go.Figure(go.Funnel(
        y=knockout_stages,
        x=knockout_values,
        textinfo='text',
        text=['VID beat DEL by 76 runs', 'VID beat KAR by 6 wickets', 'VID beat SAUR by 38 runs', '🏆 VIDARBHA'],
        marker=dict(color=[COLORS['tertiary'], COLORS['secondary'], COLORS['primary'], COLORS['gold']])
    ))
    fig_funnel.update_layout(
        title="Vidarbha's Knockout Journey",
        font={'family': 'Arial'},
        height=400
    )
    
    # Top Performers for Vidarbha
    vid_batting = player_batting[player_batting['Team'] == 'VID']
    vid_bowling = player_bowling[player_bowling['Team'] == 'VID']
    
    top_vid_batters = vid_batting.groupby('Batter').agg({
        'Runs': 'sum', 'Balls': 'sum', '4s': 'sum', '6s': 'sum'
    }).reset_index().nlargest(5, 'Runs')
    
    top_vid_bowlers = vid_bowling.groupby('Bowler').agg({
        'Wickets': 'sum', 'Runs': 'sum', 'Overs': 'sum'
    }).reset_index().nlargest(5, 'Wickets')
    
    fig_vid_batters = px.bar(
        top_vid_batters,
        x='Runs',
        y='Batter',
        orientation='h',
        color='Runs',
        color_continuous_scale='Reds',
        text='Runs',
        labels={'Runs': 'Total Runs', 'Batter': 'Player'}
    )
    fig_vid_batters.update_layout(
        title='Top Run Scorers for Vidarbha',
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=350
    )
    fig_vid_batters.update_traces(textposition='outside')
    
    fig_vid_bowlers = px.bar(
        top_vid_bowlers,
        x='Wickets',
        y='Bowler',
        orientation='h',
        color='Wickets',
        color_continuous_scale='Greens',
        text='Wickets',
        labels={'Wickets': 'Total Wickets', 'Bowler': 'Player'}
    )
    fig_vid_bowlers.update_layout(
        title='Top Wicket Takers for Vidarbha',
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=350
    )
    fig_vid_bowlers.update_traces(textposition='outside')
    
    charts_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_funnel, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_vid_batters, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_vid_bowlers, config={'displayModeBar': False})
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=4),
    ], className='mb-4')
    
    # Match Results Timeline Table
    timeline_table = dbc.Card([
        dbc.CardHeader("Vidarbha's Complete Journey", style={'backgroundColor': COLORS['gold'], 'color': 'black', 'fontWeight': 'bold'}),
        dbc.CardBody([
            dash_table.DataTable(
                data=timeline_df.to_dict('records'),
                columns=[
                    {'name': '#', 'id': 'match_num'},
                    {'name': 'Date', 'id': 'date'},
                    {'name': 'Opponent', 'id': 'opponent'},
                    {'name': 'VID Score', 'id': 'vid_score'},
                    {'name': 'Opp Score', 'id': 'opp_score'},
                    {'name': 'Result', 'id': 'result'},
                    {'name': 'Margin', 'id': 'margin'},
                    {'name': 'Stage', 'id': 'stage'},
                ],
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial'},
                style_header={'backgroundColor': COLORS['secondary'], 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'filter_query': '{result} = Won'}, 'backgroundColor': '#d4edda'},
                    {'if': {'filter_query': '{result} = Lost'}, 'backgroundColor': '#f8d7da'},
                    {'if': {'filter_query': '{stage} = Final'}, 'backgroundColor': '#fff3cd', 'fontWeight': 'bold'},
                ]
            )
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    # Vidarbha vs Runner-up Comparison
    saur_stats = team_stats[team_stats['team'] == 'SAUR'].iloc[0]
    
    comparison_metrics = ['Win %', 'Avg Score', 'Strike Rate', 'Economy', 'Total Runs', 'Total Wickets']
    vid_values = [vid_stats['win_pct'], vid_stats['avg_score'], vid_stats['team_sr'], 
                  vid_stats['economy'], vid_stats['total_runs'], vid_stats['total_wickets']]
    saur_values = [saur_stats['win_pct'], saur_stats['avg_score'], saur_stats['team_sr'],
                   saur_stats['economy'], saur_stats['total_runs'], saur_stats['total_wickets']]
    
    fig_vs_runnerup = go.Figure()
    fig_vs_runnerup.add_trace(go.Bar(
        name='Vidarbha (Champion)',
        x=comparison_metrics,
        y=vid_values,
        marker_color=COLORS['gold'],
        text=[f'{v:.1f}' if isinstance(v, float) else str(int(v)) for v in vid_values],
        textposition='outside'
    ))
    fig_vs_runnerup.add_trace(go.Bar(
        name='Saurashtra (Runner-up)',
        x=comparison_metrics,
        y=saur_values,
        marker_color=COLORS['tertiary'],
        text=[f'{v:.1f}' if isinstance(v, float) else str(int(v)) for v in saur_values],
        textposition='outside'
    ))
    fig_vs_runnerup.update_layout(
        title='Champion vs Runner-up Comparison',
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial'},
        height=400
    )
    
    comparison_card = dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig_vs_runnerup, config={'displayModeBar': False})
        ])
    ], className='shadow-sm mb-4', style={'borderRadius': '10px', 'border': 'none'})
    
    return html.Div([
        html.H2("🏆 Champion's Journey - Vidarbha", className='mb-4', style={'color': COLORS['gold']}),
        champion_header,
        kpi_row,
        create_section_header('Tournament Progression', 'Match-by-match performance'),
        charts_row1,
        create_section_header('Knockout Stage & Top Performers', 'The road to glory'),
        charts_row2,
        timeline_table,
        create_section_header('Champion vs Runner-up', 'Final comparison'),
        comparison_card
    ])


# =============================================================================
# APP LAYOUT
# =============================================================================

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# ============================================================================
# CRITICAL FOR RENDER DEPLOYMENT - DO NOT REMOVE
# ============================================================================
server = app.server  # This line is REQUIRED for Gunicorn/Render
# ============================================================================

app.title = 'Vijay Hazare Trophy 2025-26 Dashboard'

# Sidebar
sidebar = html.Div([
    html.Div([
        html.H4('🏏 VHT 2025-26', className='text-white mb-0'),
        html.Small('Dashboard', className='text-white-50')
    ], className='p-3'),
    html.Hr(className='my-2', style={'borderColor': 'rgba(255,255,255,0.2)'}),
    dbc.Nav([
        dbc.NavLink([html.Span('🏠'), html.Span(' Overview', className='ms-2')], 
                    href='/', active='exact', className='text-white'),
        dbc.NavLink([html.Span('📊'), html.Span(' Team Comparison', className='ms-2')], 
                    href='/team-comparison', active='exact', className='text-white'),
        dbc.NavLink([html.Span('🏏'), html.Span(' Batting Analysis', className='ms-2')], 
                    href='/batting', active='exact', className='text-white'),
        dbc.NavLink([html.Span('🎯'), html.Span(' Bowling Analysis', className='ms-2')], 
                    href='/bowling', active='exact', className='text-white'),
        dbc.NavLink([html.Span('🎲'), html.Span(' Toss & Venue', className='ms-2')], 
                    href='/toss-venue', active='exact', className='text-white'),
        dbc.NavLink([html.Span('⚔️'), html.Span(' Match Situations', className='ms-2')], 
                    href='/match-situations', active='exact', className='text-white'),
        dbc.NavLink([html.Span('🧤'), html.Span(' Dismissal Patterns', className='ms-2')], 
                    href='/dismissals', active='exact', className='text-white'),
        dbc.NavLink([html.Span('🎖️'), html.Span(' Qualified Teams', className='ms-2')], 
                    href='/qualified', active='exact', className='text-white'),
        dbc.NavLink([html.Span('🏆'), html.Span(" Champion's Journey", className='ms-2')], 
                    href='/champions', active='exact', className='text-white'),
    ], vertical=True, pills=True, className='flex-column'),
], style={
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '220px',
    'backgroundColor': COLORS['secondary'],
    'padding': '0',
    'overflowY': 'auto'
})

# Content area
content = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', style={'padding': '20px'})
], style={
    'marginLeft': '220px',
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh'
})

# Main layout
app.layout = html.Div([sidebar, content])


# =============================================================================
# CALLBACKS
# =============================================================================

# Page routing callback
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/' or pathname == '':
        return create_page_overview()
    elif pathname == '/team-comparison':
        return create_page_team_comparison()
    elif pathname == '/batting':
        return create_page_batting()
    elif pathname == '/bowling':
        return create_page_bowling()
    elif pathname == '/toss-venue':
        return create_page_toss_venue()
    elif pathname == '/match-situations':
        return create_page_match_situations()
    elif pathname == '/dismissals':
        return create_page_dismissals()
    elif pathname == '/qualified':
        return create_page_qualified()
    elif pathname == '/champions':
        return create_page_champions()
    else:
        return create_page_overview()


# Standings tabs callback
@app.callback(
    Output('standings-content', 'children'),
    Input('standings-tabs', 'active_tab')
)
def update_standings(active_tab):
    group_map = {
        'group-a': 'Group A',
        'group-b': 'Group B',
        'group-c': 'Group C',
        'group-d': 'Group D',
        'plate': 'Plate'
    }
    
    group = group_map.get(active_tab, 'Group A')
    group_data = standings[standings['Group'] == group][['Pos', 'Team_Clean', 'P', 'W', 'L', 'Pts', 'NRR', 'Qualified']]
    group_data.columns = ['Pos', 'Team', 'P', 'W', 'L', 'Pts', 'NRR', 'Qualified']
    
    return dash_table.DataTable(
        data=group_data.to_dict('records'),
        columns=[
            {'name': 'Pos', 'id': 'Pos'},
            {'name': 'Team', 'id': 'Team'},
            {'name': 'P', 'id': 'P'},
            {'name': 'W', 'id': 'W'},
            {'name': 'L', 'id': 'L'},
            {'name': 'Pts', 'id': 'Pts'},
            {'name': 'NRR', 'id': 'NRR'},
        ],
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Arial'},
        style_header={'backgroundColor': COLORS['secondary'], 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'filter_query': '{Qualified} = true'}, 'backgroundColor': '#fff3cd', 'fontWeight': 'bold'},
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
        ]
    )


# Team comparison radar chart callback
@app.callback(
    Output('team-radar-chart', 'figure'),
    [Input('team-selector', 'value'),
     Input('team-compare-selector', 'value')]
)
def update_radar_chart(team1, team2):
    team1_stats = team_stats[team_stats['team'] == team1].iloc[0]
    team2_stats = team_stats[team_stats['team'] == team2].iloc[0]
    
    categories = ['Win %', 'Avg Score', 'Strike Rate', 'Economy (inv)', 'Wickets/Match', 'NRR']
    
    # Normalize values
    team1_values = [
        team1_stats['win_pct'] / 100,
        team1_stats['avg_score'] / 350,
        team1_stats['team_sr'] / 120,
        1 - (team1_stats['economy'] / 10),
        team1_stats['wickets_per_match'] / 10,
        (team1_stats['nrr'] + 3) / 6
    ]
    
    team2_values = [
        team2_stats['win_pct'] / 100,
        team2_stats['avg_score'] / 350,
        team2_stats['team_sr'] / 120,
        1 - (team2_stats['economy'] / 10),
        team2_stats['wickets_per_match'] / 10,
        (team2_stats['nrr'] + 3) / 6
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=team1_values + [team1_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=team1,
        line_color=get_team_color(team1)
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=team2_values + [team2_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=team2,
        line_color=get_team_color(team2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f'{team1} vs {team2} Comparison',
        font={'family': 'Arial'},
        height=400,
        showlegend=True
    )
    
    return fig


# Team comparison stats callback
@app.callback(
    Output('team-comparison-stats', 'children'),
    [Input('team-selector', 'value'),
     Input('team-compare-selector', 'value')]
)
def update_comparison_stats(team1, team2):
    t1 = team_stats[team_stats['team'] == team1].iloc[0]
    t2 = team_stats[team_stats['team'] == team2].iloc[0]
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(f"{team1} - {t1['team_name']}", style={'backgroundColor': get_team_color(team1), 'color': 'white', 'fontWeight': 'bold'}),
                dbc.CardBody([
                    html.P(f"Matches: {int(t1['matches_played'])} | Wins: {int(t1['wins'])} | Win%: {t1['win_pct']:.1f}%"),
                    html.P(f"Avg Score: {t1['avg_score']:.0f} | SR: {t1['team_sr']:.1f}"),
                    html.P(f"Economy: {t1['economy']:.2f} | Wickets/Match: {t1['wickets_per_match']:.1f}"),
                    html.P(f"NRR: {t1['nrr']:+.3f}", className='fw-bold')
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(f"{team2} - {t2['team_name']}", style={'backgroundColor': get_team_color(team2), 'color': 'white', 'fontWeight': 'bold'}),
                dbc.CardBody([
                    html.P(f"Matches: {int(t2['matches_played'])} | Wins: {int(t2['wins'])} | Win%: {t2['win_pct']:.1f}%"),
                    html.P(f"Avg Score: {t2['avg_score']:.0f} | SR: {t2['team_sr']:.1f}"),
                    html.P(f"Economy: {t2['economy']:.2f} | Wickets/Match: {t2['wickets_per_match']:.1f}"),
                    html.P(f"NRR: {t2['nrr']:+.3f}", className='fw-bold')
                ])
            ], className='shadow-sm', style={'borderRadius': '10px', 'border': 'none'})
        ], md=6),
    ], className='mb-4')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print('='*60)
    print('VIJAY HAZARE TROPHY 2025-26 DASHBOARD')
    print('='*60)
    print(f'\nLoaded Data:')
    print(f'  - Matches: {len(matches)}')
    print(f'  - Teams: {len(team_stats)}')
    print(f'  - Player Batting Records: {len(player_batting)}')
    print(f'  - Player Bowling Records: {len(player_bowling)}')
    print(f'\n🏆 Champion: VIDARBHA')
    print(f'\nStarting server...')
    print('='*60)
    
    # For local development only
    # In production, Gunicorn uses the 'server' variable
    app.run(debug=False, host='0.0.0.0', port=8050)
