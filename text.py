plan = '''1. Get all pitch data from 2020-08-01 to 2020-08-07:
all_pitches = statcast('2020-08-01', '2020-08-07')

2. Filter for just curveballs:
all_curves = all_pitches[all_pitches['pitch_type'] == 'CU']

3. Create a feature vector for each pitcher's curveballs:
pitcher_curves = all_curves.groupby('pitcher')
pitcher_features = pitcher_curves[['release_speed', 'release_spin', 'pfx_x', 'pfx_z']].mean().reset_index()

4. Get Max Scherzer's player ID:
from pybaseball import playerid_lookup
scherzer_id = playerid_lookup('scherzer', 'max')['key_mlbam'][0]

5. Get Scherzer's feature vector:
scherzer_features = pitcher_features[pitcher_features['pitcher'] == scherzer_id]

6. Import scikit-learn and create a knn model:
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=4)
knn.fit(pitcher_features[['release_speed', 'release_spin', 'pfx_x', 'pfx_z']])

7. Find the 3 pitchers closest to Scherzer:
distances, indices = knn.kneighbors(scherzer_features[['release_speed', 'release_spin', 'pfx_x', 'pfx_z']])
closest_indices = indices[0][1:4]
similar_pitchers = pitcher_features.iloc[closest_indices]
print(similar_pitchers[['pitcher']])'''

task =  "Consider the first week of August 2020 - find 3 pitchers who's curveballs were most similar to Max Scherzer's."

function_detail = '''Text between the <playerid_lookup_documentation></playerid_lookup_documentation> tags is documentation for the playerid_lookup function.  Consult this section to confirm which attributes to pass into the playerid_lookup function.
<playerid_lookup_documentation>
# Player ID Lookup

## Single Player Lookup

`playerid_lookup(last, first=None, fuzzy=False)`

Look up a player's MLBAM, Retrosheet, FanGraphs, and Baseball Reference ID by name.

## Arguments
`last:` String. The player's last name. Case insensitive.

`first:` String. Optional. The player's first name. Case insensitive.

`fuzzy:` Boolean. Optional. Search for inexact name matches, the 5 closest will be returned.

Providing last name only will return all available id data for players with that last name (this will return several rows for a common last name like Jones, for example.) If multiple players exist for a (last name, first name) pair, you can figure out who's who by seeing their first and last years of play in the fields `mlb_played_first` and `mlb_played_last`.

This data comes from Chadwick Bureau, meaning that there are several people in this data who are not MLB players. For this reason, supplying both last and first name is recommended to narrow your search. 

## Examples of valid queries

```python
from pybaseball import playerid_lookup

# find the ids of all players with last name Jones (returns 1,314 rows)
data = playerid_lookup('jones')

# only return the ids of chipper jones (returns one row)
data = playerid_lookup('jones','chipper')

# Will return all players named Pedro Martinez (returns *2* rows)
data = playerid_lookup("martinez", "pedro", fuzzy=True)

# Will return the 5 closest names to "yadi molina" (returns 5 rows)
# First row will be Yadier Molina
data = playerid_lookup("molina", "yadi", fuzzy=True)
```

## List Lookup

`player_search_list(player_list)`

Look up a list of player ID's by name, return a data frame of all players

`player_list:` List. A list of tuples, of the form `(last, first)`. Case Insensitive.

Sources are the same as those used in the above `playerid_lookup` function. Queries for this function must be exact name matches.

## Examples of valid queries

```python

from pybaseball import player_search_list

# Will return the ids for both Lou Brock and Chipper Jones (returns 2 rows)
data = player_search_list([("brock","lou"), ("jones","chipper")])

```
</playerid_lookup_documentation>
Text between the <playerid_lookup_dictionary></playerid_lookup_dictionary> tags is the data dictionary for the playerid_lookup function.
<playerid_lookup_dictionary>

name_last: player's last name
name_first: player's first name
key_mlbam: MLB Advanced Media ID
key_retro: MLB Retrosheet ID
key_bbref: MLB Baseball Reference ID
key_fangraphs: MLB FanGraphs ID
mlb_played_first: first season played by the player
mlb_played_last: last season played by the player
</playerid_lookup_dictionary>Text between the <statcast_pitcher_documentation></statcast_pitcher_documentation> tags is documentation for the statcast_pitcher function.  Consult this section to confirm which attributes to pass into the statcast_pitcher function.
<statcast_pitcher_documentation>
# Statcast Pitcher
`statcast_pitcher(start_dt=[yesterday's date], end_dt=None, player_id)`

The statcast function retrieves pitch-level statcast data for a given date or range or dates. 

## Arguments
`start_dt:` first day for which you want to retrieve data. Defaults to yesterday's date if nothing is entered. If you only want data for one date, supply a `start_dt` value but not an `end_dt` value. Format: YYYY-MM-DD. 

`end_dt:` last day for which you want to retrieve data. Defaults to None. If you want to retrieve data for more than one day, both a `start_dt` and `end_dt` value must be given. Format: YYYY-MM-DD. 

`player_id:` MLBAM player ID for the pitcher you want to retrieve data for. To find a player's MLBAM ID, see the function [playerid_lookup](http://github.com/jldbc/pybaseball/docs/playerid_lookup.md) or the examples below. 

### A note on data availability 
The earliest available statcast data comes from the 2008 season when the system was first introduced to Major League Baseball. Queries before this year will not work. Further, some features were introduced after the 2008 season. Launch speed angle, for example, is only available from the 2015 season forward. 

### Known issue
In rare cases where a player has seen greater than 30,000 pitches over the time period specified in your query, only the first 30,000 of these plays will be returned. There is a fix in the works for this

## Examples of valid queries

```python
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup

# find Chris Sale's player id (mlbam_key)
playerid_lookup('sale','chris')

# get all available data
data = statcast_pitcher('2008-04-01', '2017-07-15', player_id = 519242)

# get data for July 15th, 2017
data = statcast_pitcher('2017-07-15','2017-07-15', player_id = 519242)
```
</statcast_pitcher_documentation>
Text between the <statcast_pitcher_dictionary></statcast_pitcher_dictionary> tags is the data dictionary for the statcast_pitcher function.
<statcast_pitcher_dictionary>

pitch_type
The type of pitch derived from Statcast.

game_date
Date of the Game.

release_speed
Pitch velocities from 2008-16 are via Pitch F/X, and adjusted to roughly out-of-hand release point. All velocities from 2017 and beyond are Statcast, which are reported out-of-hand.

release_pos_x
Horizontal Release Position of the ball measured in feet from the catcher's perspective.

release_pos_z
Vertical Release Position of the ball measured in feet from the catcher's perspective.

player_name
Player's name tied to the event of the search formatted as "Last Name, First Name"

batter
MLB Player Id tied to the play event.

pitcher
MLB Player Id tied to the play event.

events
Event of the resulting Plate Appearance.

description
Description of the resulting pitch.

zone
Zone location of the ball when it crosses the plate from the catcher's perspective.

des
Plate appearance description from game day.

game_type
Type of Game. E = Exhibition, S = Spring Training, R = Regular Season, F = Wild Card, D = Divisional Series, L = League Championship Series, W = World Series

stand
Side of the plate batter is standing.

p_throws
Hand pitcher throws with.

home_team
Abbreviation of home team.

away_team
Abbreviation of away team.

type
Short hand of pitch result. B = ball, S = strike, X = in play.

hit_location
Position of first fielder to touch the ball.

bb_type
Batted ball type, ground_ball, line_drive, fly_ball, popup.

balls
Pre-pitch number of balls in count.

strikes
Pre-pitch number of strikes in count.

game_year
Year game took place.

pfx_x
Horizontal movement in feet from the catcher's perspective.

pfx_z
Vertical movement in feet from the catcher's perpsective.

plate_x
Horizontal position of the ball when it crosses home plate from the catcher's perspective.

plate_z
Vertical position of the ball when it crosses home plate from the catcher's perspective.

on_3b
Pre-pitch MLB Player Id of Runner on 3B.

on_2b
Pre-pitch MLB Player Id of Runner on 2B.

on_1b
Pre-pitch MLB Player Id of Runner on 1B.

outs_when_up
Pre-pitch number of outs.

inning
Pre-pitch inning number.

inning_topbot
Pre-pitch top or bottom of inning.

hc_x
Hit coordinate X of batted ball.

hc_y
Hit coordinate Y of batted ball.

sv_id
Non-unique Id of play event per game.

vx0
The velocity of the pitch, in feet per second, in x-dimension, determined at y=50 feet.

vy0
The velocity of the pitch, in feet per second, in y-dimension, determined at y=50 feet.

vy0
The velocity of the pitch, in feet per second, in z-dimension, determined at y=50 feet.

ax
The acceleration of the pitch, in feet per second per second, in x-dimension, determined at y=50 feet.

ay
The acceleration of the pitch, in feet per second per second, in y-dimension, determined at y=50 feet.

az
The acceleration of the pitch, in feet per second per second, in z-dimension, determined at y=50 feet.

sz_top
Top of the batter's strike zone set by the operator when the ball is halfway to the plate.

sz_bot
Bottom of the batter's strike zone set by the operator when the ball is halfway to the plate.

hit_distance
Projected hit distance of the batted ball.

launch_speed
Exit velocity of the batted ball as tracked by Statcast. For the limited subset of batted balls not tracked directly, estimates are included based on the process described here.

launch_angle
Launch angle of the batted ball as tracked by Statcast. For the limited subset of batted balls not tracked directly, estimates are included based on the process described here.

effective_speed
Derived speed based on the the extension of the pitcher's release.

release_spin
Spin rate of pitch tracked by Statcast.

release_extension
Release extension of pitch in feet as tracked by Statcast.

game_pk
Unique Id for Game.

pitcher
MLB Player Id tied to the play event.

fielder_2
MLB Player Id for catcher.

fielder_3
MLB Player Id for 1B.

fielder_4
MLB Player Id for 2B.

fielder_5
MLB Player Id for 3B.

fielder_6
MLB Player Id for SS.

fielder_7
MLB Player Id for LF.

fielder_8
MLB Player Id for CF.

fielder_9
MLB Player Id for RF.

release_pos_y
Release position of pitch measured in feet from the catcher's perspective.

estimated_ba_using_speedangle
Estimated Batting Avg based on launch angle and exit velocity.

estimated_woba_using_speedangle
Estimated wOBA based on launch angle and exit velocity.

woba_value
wOBA value based on result of play.

woba_denom
wOBA denominator based on result of play.

babip_value
BABIP value based on result of play.

iso_value
ISO value based on result of play.

launch_speed_angle
Launch speed/angle zone based on launch angle and exit velocity.
1: Weak
2: Topped
3: Under
4: Flare/Burner
5: Solid Contact
6: Barrel


at_bat_number
Plate appearance number of the game.

pitch_number
Total pitch number of the plate appearance.

pitch_name
The name of the pitch derived from the Statcast Data.

home_score
Pre-pitch home score

away_score
Pre-pitch away score

bat_score
Pre-pitch bat team score

fld_score
Pre-pitch field team score

post_home_score
Post-pitch home score

post_away_score
Post-pitch away score

post_bat_score
Post-pitch bat team score

if_fielding_alignment
Infield fielding alignment at the time of the pitch.

of_fielding_alignment
Outfield fielding alignment at the time of the pitch.

spin_axis
The Spin Axis in the 2D X-Z plane in degrees from 0 to 360, such that 180 represents a pure backspin fastball and 0 degrees represents a pure topspin (12-6) curveball

delta_home_win_exp
The change in Win Expectancy before the Plate Appearance and after the Plate Appearance

delta_run_exp
The change in Run Expectancy before the Pitch and after the Pitch

</statcast_pitcher_dictionary>Text between the <statcast_documentation></statcast_documentation> tags is documentation for the statcast function.  Consult this section to confirm which attributes to pass into the statcast function.
<statcast_documentation>
# Statcast
`statcast(start_dt=[yesterday's date], end_dt=None, team=None, verbose=True, parallel=True)`

The `statcast` function retrieves pitch-level statcast data for a given date or range or dates. 

## Returned data
This function returns a pandas `DataFrame` with one entry for each pitch in the
query. The data returned for each pitch is explained on
[Baseball Savant](https://baseballsavant.mlb.com/csv-docs).

## Arguments
`start_dt:` first day for which you want to retrieve data. Defaults to yesterday's date if nothing is entered. If you only want data for one date, supply a `start_dt` value but not an `end_dt` value. Format: YYYY-MM-DD. 

`end_dt:` last day for which you want to retrieve data. Defaults to None. If you want to retrieve data for more than one day, both a `start_dt` and `end_dt` value must be given. Format: YYYY-MM-DD. 

`team:` optional. If you only want statcast data for one team, supply that team's abbreviation here (i.e. BOS, SEA, NYY, etc).

`verbose:` Boolean, default=True. If set to True this will provide updates on query progress, if set to False it will not. 

`parallel:` Boolean, default=True. Whether to parallelize HTTP requests in large queries.

### A note on data availability 
The earliest available statcast data comes from the 2008 season when the system was first introduced to Major League Baseball. Queries before this year will not work. Further, some features were introduced after the 2008 season. Launch speed angle, for example, is only available from the 2015 season forward. 

### A note on query time
Baseball savant limits queries to 30000 rows each. For this reason, if your request is for a period of greater than 5 days, it will be broken into two or more smaller requests. The data will still be returned to you in a single dataframe, but it will take slightly longer. 

### A note on parallelization
Large queries with requests made in parallel complete substantially faster. This option exists to accommodate compute environments where multiprocessing is disabled (e.g. some AWS Lambda environments).

## Examples of valid queries

```python
from pybaseball import statcast

# get all statcast data for July 4th, 2017
data = statcast('2017-07-04')

#get data for the first seven days of August in 2016
data = statcast('2016-08-01', '2016-08-07')

#get all data for the Texas Rangers in the 2016 season
data = statcast('2016-04-01', '2016-10-30', team='TEX')

# get data for yesterday
data = statcast()```

</statcast_documentation>
Text between the <statcast_dictionary></statcast_dictionary> tags is the data dictionary for the statcast function.
<statcast_dictionary>

pitch_type
The type of pitch derived from Statcast.

game_date
Date of the Game.

release_speed
Pitch velocities from 2008-16 are via Pitch F/X, and adjusted to roughly out-of-hand release point. All velocities from 2017 and beyond are Statcast, which are reported out-of-hand.

release_pos_x
Horizontal Release Position of the ball measured in feet from the catcher's perspective.

release_pos_z
Vertical Release Position of the ball measured in feet from the catcher's perspective.

player_name
Player's name tied to the event of the search formatted as "Last Name, First Name"

batter
MLB Player Id tied to the play event.

pitcher
MLB Player Id tied to the play event.

events
Event of the resulting Plate Appearance.

description
Description of the resulting pitch.

zone
Zone location of the ball when it crosses the plate from the catcher's perspective.

des
Plate appearance description from game day.

game_type
Type of Game. E = Exhibition, S = Spring Training, R = Regular Season, F = Wild Card, D = Divisional Series, L = League Championship Series, W = World Series

stand
Side of the plate batter is standing.

p_throws
Hand pitcher throws with.

home_team
Abbreviation of home team.

away_team
Abbreviation of away team.

type
Short hand of pitch result. B = ball, S = strike, X = in play.

hit_location
Position of first fielder to touch the ball.

bb_type
Batted ball type, ground_ball, line_drive, fly_ball, popup.

balls
Pre-pitch number of balls in count.

strikes
Pre-pitch number of strikes in count.

game_year
Year game took place.

pfx_x
Horizontal movement in feet from the catcher's perspective.

pfx_z
Vertical movement in feet from the catcher's perpsective.

plate_x
Horizontal position of the ball when it crosses home plate from the catcher's perspective.

plate_z
Vertical position of the ball when it crosses home plate from the catcher's perspective.

on_3b
Pre-pitch MLB Player Id of Runner on 3B.

on_2b
Pre-pitch MLB Player Id of Runner on 2B.

on_1b
Pre-pitch MLB Player Id of Runner on 1B.

outs_when_up
Pre-pitch number of outs.

inning
Pre-pitch inning number.

inning_topbot
Pre-pitch top or bottom of inning.

hc_x
Hit coordinate X of batted ball.

hc_y
Hit coordinate Y of batted ball.

sv_id
Non-unique Id of play event per game.

vx0
The velocity of the pitch, in feet per second, in x-dimension, determined at y=50 feet.

vy0
The velocity of the pitch, in feet per second, in y-dimension, determined at y=50 feet.

vy0
The velocity of the pitch, in feet per second, in z-dimension, determined at y=50 feet.

ax
The acceleration of the pitch, in feet per second per second, in x-dimension, determined at y=50 feet.

ay
The acceleration of the pitch, in feet per second per second, in y-dimension, determined at y=50 feet.

az
The acceleration of the pitch, in feet per second per second, in z-dimension, determined at y=50 feet.

sz_top
Top of the batter's strike zone set by the operator when the ball is halfway to the plate.

sz_bot
Bottom of the batter's strike zone set by the operator when the ball is halfway to the plate.

hit_distance
Projected hit distance of the batted ball.

launch_speed
Exit velocity of the batted ball as tracked by Statcast. For the limited subset of batted balls not tracked directly, estimates are included based on the process described here.

launch_angle
Launch angle of the batted ball as tracked by Statcast. For the limited subset of batted balls not tracked directly, estimates are included based on the process described here.

effective_speed
Derived speed based on the the extension of the pitcher's release.

release_spin
Spin rate of pitch tracked by Statcast.

release_extension
Release extension of pitch in feet as tracked by Statcast.

game_pk
Unique Id for Game.

pitcher
MLB Player Id tied to the play event.

fielder_2
MLB Player Id for catcher.

fielder_3
MLB Player Id for 1B.

fielder_4
MLB Player Id for 2B.

fielder_5
MLB Player Id for 3B.

fielder_6
MLB Player Id for SS.

fielder_7
MLB Player Id for LF.

fielder_8
MLB Player Id for CF.

fielder_9
MLB Player Id for RF.

release_pos_y
Release position of pitch measured in feet from the catcher's perspective.

estimated_ba_using_speedangle
Estimated Batting Avg based on launch angle and exit velocity.

estimated_woba_using_speedangle
Estimated wOBA based on launch angle and exit velocity.

woba_value
wOBA value based on result of play.

woba_denom
wOBA denominator based on result of play.

babip_value
BABIP value based on result of play.

iso_value
ISO value based on result of play.

launch_speed_angle
Launch speed/angle zone based on launch angle and exit velocity.
1: Weak
2: Topped
3: Under
4: Flare/Burner
5: Solid Contact
6: Barrel


at_bat_number
Plate appearance number of the game.

pitch_number
Total pitch number of the plate appearance.

pitch_name
The name of the pitch derived from the Statcast Data.

home_score
Pre-pitch home score

away_score
Pre-pitch away score

bat_score
Pre-pitch bat team score

fld_score
Pre-pitch field team score

post_home_score
Post-pitch home score

post_away_score
Post-pitch away score

post_bat_score
Post-pitch bat team score

if_fielding_alignment
Infield fielding alignment at the time of the pitch.

of_fielding_alignment
Outfield fielding alignment at the time of the pitch.

spin_axis
The Spin Axis in the 2D X-Z plane in degrees from 0 to 360, such that 180 represents a pure backspin fastball and 0 degrees represents a pure topspin (12-6) curveball

delta_home_win_exp
The change in Win Expectancy before the Plate Appearance and after the Plate Appearance

delta_run_exp
The change in Run Expectancy before the Pitch and after the Pitch

</statcast_dictionary>'''