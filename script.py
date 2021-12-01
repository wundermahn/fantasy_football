# Import Statements

# Sleeper wrapper
from sleeper_wrapper import League, Stats

# Progress Bar
from tqdm import tqdm

# Misc
import requests, pandas as pd, numpy as np, warnings as warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# Find your league
league = League(728718179756519424)

# Create look up dictionary for roster IDs
lookup_dict = {1: 'Robby', 2: 'Matt', 3: 'Tony', 4: 'Tim', 5: 'Kade', 6: 'BJ', 7: 'Erik', 8: 'Joey', 9: 'Ric', 10: 'Chris'}

# Create roster requirements dictionary
requirements_dict = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DEF': 1, 'FLEX': 2}

# Which week to run program for
week = 12

# Current records
id_dict = {
    '449627600818532352': 'Robby',
    '600507598768246784': 'Tony',
    '732818307723411456': 'Kade',
    '736993651930042368': 'Matt',
    '737006160510615552': 'Tim',
    '737054798611943424': 'BJ',
    '737061957353472000': 'Erik',
    '737318822301868032': 'Joey',
    '737476504782557184': 'Ric',
    '737732549224357888': 'Chris'
    }

# Dictionary of current team names and the manager
team_name_dict = dict((value, id_dict.get(key)) for key, value in league.map_users_to_team_name(league.get_users()).items())    

# List of managers
names = list(set([v for k,v in team_name_dict.items()]))
## TODO: Automate retrieving this

# Function to parse the players json and turn it into a dataframe
def parse_players(players_json):
    # Create lists for data that we care about
    last_names = []
    first_names = []
    positions = []
    ids = []

    # Loop through the dictionary
    for k,v in tqdm(players_json.items()):
        # Create a temp dict
        item = players_json[k]
        # Save the metadata we carry about
        last_names.append(item['last_name'])
        first_names.append(item['first_name'])
        positions.append(item['position'])
        ids.append(item['player_id'])
    
    # Create a list of full/whole names
    names = ["{} {}".format(x,y) for x,y in zip(first_names, last_names)]

    # Convert to dataframe
    res_df = pd.DataFrame({'Name': names, 'First Name': first_names, 'Last Name': last_names, 'Position': positions, 'ID': ids})
    
    # Return it
    return res_df

# Get the stats for a list of player IDs
def get_stats(players: list, year: int, week: int) -> list:
    # Create stats object
    stats = Stats()
    # Get weekly stats
    week_stats = stats.get_week_stats("regular", year, week)
    # Return scores for each player
    scores = [stats.get_player_week_score(week_stats, player) for player in players]
    
    return scores    

# Function to retrieve data about the league
def get_league_data(df, players_df):

    # Drop rows where user did not set player
    # df.drop(df.loc[df['starters']=='0'].index, inplace=True)
    df = df.query("starters != '0'")

    winners = []
    losers = []
    scores = {}
    games = {}

    # Go throughe ach matchup for the week
    for matchup in df['matchup_id'].unique():        
        
        # Limit the data to just the current matchup
        matchup_df = df.loc[df['matchup_id']==matchup]

        # Compare the two players in the matchup
        a = [matchup_df['roster_id'].unique()[0], matchup_df['points'].unique()[0]]
        b = [matchup_df['roster_id'].unique()[1], matchup_df['points'].unique()[1]]

        scores[a[0]] = a[1]
        scores[b[0]] = b[1]

        games[matchup] = a[1]+b[1]

        # Format strings
        if a[1] > b[1]:
            w = "{} (vs {})".format(a[0], b[0])
            l = "{} (vs {})".format(b[0], a[0])
            winners.append(w)
            losers.append(l)
        elif a[1] < b[1]:
            w = "{} (vs {})".format(b[0], a[0])
            l = "{} (vs {})".format(a[0], b[0])        
            winners.append(w)
            losers.append(l)            
        else:
            w = "{} tied {}!".format(a[0], b[0])
            l = w
            winners.append(w)
            losers.append(l)            
    
    # Find top scorers, games, etc.
    top_scorer = max(scores.items(), key = lambda k : k[1])
    low_game = list(min(games.items(), key = lambda k : k[1]))
    top_game = list(max(games.items(), key = lambda k : k[1]))
    top_game[0] = list(df.loc[df['matchup_id']==top_game[0]]['roster_id'].unique())
    low_game[0] = list(df.loc[df['matchup_id']==low_game[0]]['roster_id'].unique())

    # Add in positions and player names to dataframe
    df['Position'] = [players_df.loc[players_df['ID']==id]['Position'].unique()[0] for id in df['starters']]
    df['Player Name'] = [players_df.loc[players_df['ID']==id]['Name'].unique()[0] for id in df['starters']]   

    # Get the highest scorers by position
    highest_df = df[['Position', 'starters_points', 'Player Name', 'roster_id']].sort_values("starters_points", ascending = False).groupby("Position", as_index=False).first()
    highest_df.columns = ['Position', 'Points', 'Name', 'Team']

    # Return relevant data
    return df, "WINNERS: {}".format(', '.join(winners)), "LOSERS: {}".format(', '.join(losers)), "Highest Scorer: {}".format(list(top_scorer)), \
        "Highest Scoring Game: {}".format(list(top_game)), "Lowest Scoring Game (Boo!): {}".format(list(low_game)), highest_df

# Function to compare the points of two teams
def compare_points(df, players_df, requirements, year, week):

    # Calculate weekly scores for starters
    starters = [str(x) for x in df['starters']]
    starters_names = [players_df.loc[players_df['ID']==id]['Name'].to_list()[0] for id in starters]
    starters_positions = [players_df.loc[players_df['ID']==id]['Position'].to_list()[0] for id in starters]
    starters_df = pd.DataFrame({'Player':starters_names, 'Position': starters_positions, 'Score': get_stats(starters, year, week)})
    starters_df.fillna(0, inplace=True)
    starters_df['Score'] = [x['pts_half_ppr'] if x != 0 else 0 for x in starters_df['Score']]
    starters_df.fillna(0, inplace=True)
    starters_df['Type'] = 'Starter'

    # Get total starter points
    starter_points = starters_df['Score'].sum()

    # Calculate scores for all roster spots
    _all = [x for x in df['players'].to_list()[0]]
    all_names = [players_df.loc[players_df['ID']==id]['Name'].to_list()[0] for id in _all]
    all_positions = [players_df.loc[players_df['ID']==id]['Position'].to_list()[0] for id in _all]
    all_df = pd.DataFrame({'Player':all_names, 'Position': all_positions, 'Score': get_stats(_all, year, week)})
    all_df.fillna(0, inplace=True)
    all_df['Score'] = [x['pts_half_ppr'] if x != 0 else 0 for x in all_df['Score']]
    all_df.fillna(0, inplace=True)

    # Organize by top scorers by position
    all_points = all_df.sort_values("Score", ascending = False).groupby("Position", as_index=False).head(n=1000)

    # Use the requirements dictionary of roster makeup to calculate the optimal roster based on points scored
    FLEX = requirements['FLEX']
    select_players = lambda x: x.nlargest(requirements[x.name])

    idx_best = all_points.groupby('Position')['Score'].apply(select_players).index.levels[1]
    idx_flex = all_points.loc[all_points.index.difference(idx_best), 'Score'].nlargest(FLEX).index

    # out = all_points.loc[idx_best.union(idx_flex)].sort_values('Score', ascending=False)
    max_points = all_points.loc[idx_best.union(idx_flex)].sort_values('Score', ascending=False)['Score'].sum()

    # Return a list containing 0 == max possible points, and 1 == points scored
    return [float(max_points), float(starter_points)]         

# Get weekly matchup data
def get_weekly_data(league, week: int, lookup_dict):
    # Normalize the returned json string
    weekly_df = pd.json_normalize(league.get_matchups(week)).explode(['starters', 'starters_points'])
    # Pandas magic
    weekly_df = weekly_df[weekly_df.columns.drop(list(weekly_df.filter(regex="players_points.")))]
    weekly_df.drop('custom_points', inplace=True, axis=1)
    weekly_df['roster_id'] = weekly_df['roster_id'].map(lookup_dict)    

    # Return df of weekly matchups
    return weekly_df    

# Run simulations of team performance
def run_sim(min_a, max_a, min_b, max_b, std_a=0, std_b=0, runs=1000):
    
    # Create Scaling objects based on minimum and best case team performance
    mm_a = MinMaxScaler(feature_range=(min_a, max_a))
    mm_b = MinMaxScaler(feature_range=(min_b, max_b))

    # Keep track of wins
    a_wins = 0
    b_wins = 0

    # For the specified number of epochs
    for _ in tqdm(range(runs)):
        # Create normal distributions of 1000 samples with standard deviations of the week-to-week performance of teams
        res_a = np.random.normal(size=1000, scale=std_a)
        res_b = np.random.normal(size=1000, scale=std_b)

        # Now scale that data
        res_a = mm_a.fit_transform(res_a.reshape(-1,1))
        res_b = mm_b.fit_transform(res_b.reshape(-1,1))

        # Count the number of times 
        count_a = 0
        count_b = 0

        # Keep track of who won the "games"
        for x,y in zip(res_a, res_b):
            if x > y:
                count_a += 1
            elif x < y:
                count_b += 1
            else:
                continue
        
        # Count wins
        if count_a > count_b:
            a_wins += 1
        else:
            b_wins += 1
    
    # Determine who won after all simulations
    if a_wins > b_wins:
        return 0
    else:
        return 1

# For a given week, get the [total_points_possible, scored_points] for all teams in the league
def get_points(league, week, players_df, requirements, lookup):
    # Create a dictionary of points
    points = {}

    # Get weekly data
    weekly_data = get_weekly_data(league, week, lookup)
    league_df, winners, losers, scores, high, low, positions = get_league_data(weekly_data, players_df)

    # Determine the most possible and actual scored points for each team
    for team in league_df['roster_id'].unique():
        temp = league_df.loc[league_df['roster_id']==team]
        res = compare_points(temp, players_df, requirements, 2021, week)
        points[team] = res

    # Return that dictionary
    return points        

# Wrapper function for running the code and getting weekly stats + games vs median
def wrapper(league, week, players_df, requirements, lookup, records):
    
    points = get_points(league, week, players_df, requirements, lookup)
    
    # Process against median
    median_points = np.median([v[1] for k,v in points.items()])
    # print("Median points for week {} was {}".format(week, median_points))
    for k,v in points.items():
        if v[1] > median_points:
            records.get(k)[0] += 1
        else:
            records.get(k)[1] += 1

    return records, points

def wrapper_simulations(league, week, players_df, requirements_dict, lookup_dict, records, df):
    # Collect the points for this particular week
    weekly_points = get_points(league, week, players_df, requirements_dict, lookup_dict)    

    # Collect the standard deviations for the current week
    standard_deviation_dict = {}
    for name in names:
        name_scores = []
        for k,v in scores.items():
            name_scores.append(v.get(name)[1])
        
        standard_deviation_dict[name] = np.std(name_scores) 

    # Go through each matchup for the week
    for matchup in df['matchup_id'].unique():        
        
        # Limit the data to just the current matchup
        matchup_df = df.loc[df['matchup_id']==matchup]

        a = matchup_df['roster_id'].unique()[0]
        b = matchup_df['roster_id'].unique()[1]

        max_a = weekly_points.get(a)[0]
        min_a = weekly_points.get(a)[1]

        # If the highest possible score was played, edit it it slightly
        if min_a == max_a:
            min_a -= 0.000001

        max_b = weekly_points.get(b)[0]
        min_b = weekly_points.get(b)[1]
        
        # If the highest possible score was played, edit it it slightly
        if min_b == max_b:
            min_b -= 0.000001

        std_a = standard_deviation_dict.get(a)
        std_b = standard_deviation_dict.get(b)

        sim_result = run_sim(min_a, max_a, min_b, max_b, std_a, std_b, runs=1000)

        # If the first team won, increase their wins, keep their losses the same. Keep second teams wins the same, increase their losses
        if sim_result == 0:
            records.get(a)[0]+=1
            records.get(a)[1]+=0
            records.get(b)[0]+=0
            records.get(b)[1]+=1
        # Otherwise, reverse
        else:
            records.get(a)[0]+=0
            records.get(a)[1]+=1
            records.get(b)[0]+=1
            records.get(b)[1]+=0            
    
    return records    

# Query sleeper API for list of all players
result_json_string = requests.get("https://api.sleeper.app/v1/players/nfl");

# Try to query
try:
    result_json_string.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(e)

# Convert to json
players_json = result_json_string.json()

# Get the players dataframe
players_df = parse_players(players_json)

# weekly_df = get_weekly_data(league, week, lookup_dict)
weekly_df = get_weekly_data(league, week, lookup_dict)

# Get the weekly league data print out
df, winners, losers, scores, high, low, positions = get_league_data(weekly_df, players_df)

print_str = """

WINNERS: {} \n
LOSERS: {} \n
Highest Scorer: {} \n
Highest Scoring Game: {} \n
Lowest Scoring Game (Boo!): {} \n
Top Scoring Players by Position
-------------------------------------
{}


""".format(winners, losers, scores, high, low, positions)

print(print_str)

# Get the current records for each team
records = {}
for entry in league.get_standings(league.get_rosters(), league.get_users()):
    records[team_name_dict.get(entry[0])] = [int(entry[1]), int(entry[2])]

# From the beginning of the season to the current week, grab current record (+ games against median), and the [maximum_possible_points, total_scored_points] for each team each week
scores = {}
for i in range(0,week):
    week = i+1
    records, points = wrapper(league, week, players_df, requirements_dict, lookup_dict, records)
    scores[week] = points

# Run monte carlo simulations for the current week to determine if a statistically unlucky win/loss has occurred
records = wrapper_simulations(league, week, players_df, requirements_dict, lookup_dict, records, df)

# Create a standings dataframe for the power standings
names = [k for k,v in records.items()]
wins = [v[0] for k,v in records.items()]
losses = [v[1] for k,v in records.items()]

res = pd.DataFrame({'Manager': names, 'Wins': wins, 'Losses': losses}).sort_values(by='Wins', ascending=False)
res['Ranking'] = [i+1 for i in range(res.shape[0])]

# Create a print string for the power rankings
pr_str = """

Power Rankings: Week {}

These rankings are based off of actual performance, performance vs the median, and a monte carlo simulation based on team performance thus far this season
-------------------------------------
{}
""".format(week+1, res.to_string(index=False))

print(pr_str)