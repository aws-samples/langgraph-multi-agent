statcast = '''
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
'''

player_id_lookup = '''
name_last: player's last name
name_first: player's first name
key_mlbam: MLB Advanced Media ID
key_retro: MLB Retrosheet ID
key_bbref: MLB Baseball Reference ID
key_fangraphs: MLB FanGraphs ID
mlb_played_first: first season played by the player
mlb_played_last: last season played by the player'''

pitching_stats = '''ERA: Earned Run Average
WHIP: Walks and Hits per Innings Pitched
FIP: Fielding Independent Pitching
xFIP: Expected Fielding Independent Pitching
SIERA: Skill-Interactive ERA
tERA: True Runs Allowed
K/9: Strikeout rate
BB/9: Walk rate
K%: Strikeout percentage
BB%: Walk percentage
K/BB: Strikeout-to-Walk ratio
LD%: Line drive rate
GB%: Ground ball rate
FB%: Fly ball rate
HR/FB: Home runs per fly ball rate
BABIP: Batting Average on Balls In Play
LOB%: Left On Base percentage
ERA-: ERA Minus
FIP-: FIP Minus
xFIP-: xFIP Minus
SD: Shutdowns
MD: Meltdowns
O-Swing%: Outside-the-zone swing rate
Z-Swing%: Inside-the-zone swing rate
Swing%: Swing rate
O-Contact%: Outside-the-zone contact percentage
Z-Contact%: Inside-the-zone contact percentage
Contact%: Contact percentage
Zone%: Percentage of pitches within the zone
F-Strike%: First-pitch strike percentage
SwStr%: Swinging Stike percentage
wFB: Fastball runs above average
wSL: Slider runs above average
wCT: Cutter runs above average
wCB: Curveball runs above average
wCH: Change-up runs above average
wSF: Split-finger fastball runs above average
wKN: Knuckleball runs above average
wFB/C: Fastball runs above average per 100 pitches
wSL/C: Slider runs above average per 100 pitches
wCT/C: Cutter runs above average per 100 pitches
wCB/C: Curveball runs above average per 100 pitches
wCH/C: Change-up runs above average per 100 pitches
wSF/C: Slit-fingered fastball runs above average per 100 pitches
wKN/C: Knuckleball runs above average per 100 pitches'''

batting_stats = '''
OBP: On-Base Percentage
OPS: On-base Plus Slugging
OPS+: On-base Plus Slugging Plus
wOBA: Weighted On-Base Average
wRAA: Weighted Runs Above Average
UBR: Ultimate Base Running
wRC: Weighted Runs Created
wRC+: Weighted Runs Created Plus
BABIP: Batting Average on Ball In Play
ISO: Isolated Power
HR/FB: Home Runs per Fly Ball rate
Spd: Speed Score
GB%: Ground ball percentage
FB%: Fly ball percentage
LD%: Line drive percentate
K%: Stikeout rate
BB%: Walk rate
O-Swing%: Outside-the-zone swing rate
Z-Swing%: Inside-the-zone swing rate
Swing%: Swing rate
O-Contact%: Outside-the-zone contact percentage
Z-Contact%: Inside-the-zone contact percentage
Contact%: Contact percentage
Zone%: Percentage of pitches within the zone
F-Strike%: First-pitch strike percentage
SwStr%: Swinging Stike percentage
wFB: Fastball runs above average
wSL: Slider runs above average
wCT: Cutter runs above average
wCB: Curveball runs above average
wCH: Change-up runs above average
wSF: Split-finger fastball runs above average
wKN: Knuckleball runs above average
wFB/C: Fastball runs above average per 100 pitches
wSL/C: Slider runs above average per 100 pitches
wCT/C: Cutter runs above average per 100 pitches
wCB/C: Curveball runs above average per 100 pitches
wCH/C: Change-up runs above average per 100 pitches
wSF/C: Slit-fingered fastball runs above average per 100 pitches
wKN/C: Knuckleball runs above average per 100 pitches'''

schedule_and_record = '''Date: date the game was played
Tm: team
Home_Away: indicator of Home (Home) or Away (@) game
Opp: opponenet
W/L: Win or Loss indicator
R: runs scored
RA: runs allowed
Inn: innings played
W-L: team's win-loss record after the game end
Rank: team's ranking in the division
GB: how many games back (or up) the team is in the divion
Win: winning pitcher
Loss: losing pitcher
Save: saving pitcher
Time: game duration
D/N: day or night game indicator
Attendance: game attendance
cLI: Championship Leverage Index. This is a statistic that measures the importance of winning a game to a team's chances of winning the World Series.
Streak: how many consecutive wins or losses for the team.  Consecutive wins are represented with positive values while consecutive losses are represented with negative values
Orig. Scheduled: date the game was originally scheduled (if applicable)'''

standings = '''Tm: team
W: wins
L: losses
W-L%: win-loss percentage
GB: games back in the division'''