# Moneyballing Fantasy Premier League

## Abstract

The aim of this project was to create tools to help managers playing 'Fantasy Premier League', an online game that tracks the real-world Premier League football season in England.

By collecting extensive amounts of data pertaining to both the matches themselves as well as FPL manager decisions (e.g. line-level detail for over 10,000 shots taken across over 400 games), I was able to create a series of interactive dashboards, which would provide significant guidance for FPL players.

However, the project's aim of creating models to explicitly predict player performance in future matches was not successful. Despite trying many different algorithms with cross-validated hyperparameter optimisation, trained models failed to generalise effectively to testing data.

This highlights the intrinsic 'randomness' of football - a sport that has long been difficult to model successfully.


## Table of contents

   * [The Motivation](#the-motivation)
       * [An Introduction to FPL](#an-introduction-to-fpl)
       * [Differential Players](#differential-players)
   * [The Technologies Used](#the-technologies-used)
   * [Data Gathering](#data-gathering)
       * [Data Sources](#the-data-sources)
       * [Munging the Data](#munging-the-data)
       * [Refreshing Data](#refreshing-data)
   * [Observations](#observations)
       * [Where Do FPL Points Come From?](#where-do-fpl-points-come-from)
       * [Network Analysis](#network-analysis)
   * [Goal Scoring](#thinking-about-goal-scoring)
       * [Shot Quality](#shot-quality)
       * [Expected Goals](#expected-goals)
   * [Modelling](#modelling)
       * [Building the Dataset](#building-the-dataset)
       * [Training Predictive Models](#training-predictive-models)
   * [Dashboard Building](#dashboard-building)


## The Motivation

### An Introduction to FPL

Fantasy Premier League (FPL) is an online game that follows the real-life Premier League Football season. Given a virtual budget of £100m, each ‘manager’ has to assemble a team of players, who score FPL points based on their performance in real life games.

Famous players who scored highly in previous seasons are more expensive. Therefore, good FPL managers are those able to spot cheap, unknown players with the potential to score big.

![FPLTeam](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/FPLTeam2.png)

Managers can swap one player from their team every week to bring in a replacement (who may be in better form, or have an easier run of fixtures).

A full list of the rules can be found on the FPL website: https://fantasy.premierleague.com/help/rules

### 'Differential Players'

The most sought-after players in FPL are known as 'differentials'. Differential players are those owned by a small number of FPL managers. Consistently picking ‘differentials’ is key to FPL success – if you own a high-scoring player, this isn’t overly advantageous if the same player is owned by 50% of managers.

Data confirms that there are such ‘differential’ players in the game (though they are in the minority, often making them hard to spot).

![Differentials](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/Differentials.png)

Of course, every football match generates a lot of data – both in terms of player performance and FPL manager decisions. The aim of this project, therefore, is to create a suite of tools that help managers make better picks throughout the season (namely picking better differentials), based on the data.


## The Technologies Used

* Requests and Splinter for API calls and webscraping
* Pandas for data munging
* Matplotlib and Seaborn for data visualisation
* Tableau for Dashboard generation
* Scikit-learn for Decision Trees / Random Forests
* XGboost for ensemble method learning
* GridsearchCV for hyperparameter tuning
* Keras for deep learning


## Data Gathering

### The Data Sources

The data for this project comes from two places:

1. **The official FPL API.** This gives us a lot 'first principles' data, such as basic data about each team and player in the game, as well as information about the FPL game itself, such as the number of managers choosing each player each week, and the number of FPL points that each player scores.

2. **Scraping the Premier League website.** We are able to scrape commentary (as well as some other aggregated data) straight from the Premier League website. This gives us a detailed view of every event that happened in every game. A full explanation of how this data was scraped can be found here: https://towardsdatascience.com/improve-your-data-wrangling-with-object-oriented-programming-914d3ebc83a9

**An example of FPL commentary**
![FPLTeam](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/FPLCommentary.png)

All of the data was uploaded into an SQLite database.

### Munging The Data

Getting the data from its initial string-based form into something we could actually use required significant data wrangling. In particular, we utilised OOP, creating 'Match' and 'Event' classes.

By instantiating each commentary text string as an 'Event' object, we were able to extract key information:

* What time in the game did the event happen?
* Which players were involved
* What was the outcome of the event
* etc.

![FPLTeam](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/EventObject.png)

Building back up, we were able to instantiate a 'Match' object from a dictionary of scraped data, then perform methods on that object to extract full tables of stats.

![FPLTeam](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/MatchTable.png)

A full exploration of this process can be found here: https://towardsdatascience.com/improve-your-data-wrangling-with-object-oriented-programming-914d3ebc83a9

### Refreshing Data

It's worth noting that the SQL tables can be refreshed with new data when more matches are played by running the appropriate functions in the data gathering notebook.

These new data can then be used in the other notebooks as required (e.g to generate data for the Tableau dashboards - more on these later).


## Observations

Having scraped data for over 10,000 shots across over 400 matches, we can start to see some patterns emerging.

### Where Do FPL Points Come From?

If the aim of the analysis is to maximise the number of FPL points scored by a manager's fantasy team, it's a good idea to see where these points come from.

The first thing to note is that the share of points earned by position reflects the number of players of each position. Therefore there's no intrinsic reason to favour one position over another when selecting a team (besides, there are minimum and maximum numbers of players of each position that a team can have).

![PointsByPosition](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/PointsByPosition.png)

More than half of FPL points are generated by ‘Minutes Played’, i.e. players simply being selected to play. Keeping clean sheets and scoring goals are the most common ways to generate extra points - thus FPL managers need to be mindful of both attack and defence.

![PointsByAction](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/PointsByAction.png)

### Network Analysis

We can use our shots data to identify 'important' players.

For example, we can see what share of their teams' shots (and goals) each player has been involved in.

![ShareOfShots](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/ShareOfShots.png)

And we can look at the pairs of players who have assisted each other's shots the most:

![PlayerPairs](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/PlayerPairs.png)

#### Team Graphs

We can get more scientific. Let’s think of each team as its own ‘social network’. Then we can think of each player as a node within the network for their respective teams. The weight of the ‘edges’ between each pair of nodes can then be defined by the number of times that those two players have assisted each other’s shots.

So for Manchester City, the league's highest goalscorers so far, such a network looks like this:

![MCFCNetwork](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/NetworkMC.png)

And for Wolves, who have a lot of prolific player pairs, we can see a tight 'sub-graph' of attacking players (as expected):

![WolvesNetwork](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/NetworkWO.png)

#### Node Centrality

Of course, just eyeballing a visualisation isn’t especially rigorous. We can instead use ‘weighted betweenness’ to enumerate how ‘central’ a node is within a network (and hence how important the player is to their team).

In a given network, betweenness quantifies the number of times a node acts as a bridge along the shortest path between two other nodes. In other words, to calculate Kevin De Bruyne’s ‘betweenness’, we calculate the shortest possible ‘journeys’ between every pair of players in the Manchester City network. We then calculate the share of those ‘journeys’ that pass through the De Bruyne node.

This betweenness metric can account for edge weights. In particular, we can say that the edge between two players who assist each other a lot is ‘shorter’ than other edges. The shortest path between a given pair of nodes is, of course, likelier to use these short edges, therefore players with high goal involvement will feature on more of these shortest paths, and have higher levels of betweenness.

*For the purposes of these calculations, we can say that the length of the edge between two players is the inverse of (i.e. 1 divided by) the number of times that they assisted each other. The ‘shortest path’ between each node pair is then calculated using Dijkstra’s Algorithm.*

We can then use this to see who the most 'important' players in the league are...

![PlayerImportance](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/PlayerImportance.png)

... and who the most important players are for each team.

![Talismen](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/Talismen.png)

A more detailed exploration of this analysis can be found here: https://towardsdatascience.com/who-is-the-premier-leagues-most-important-player-4f184f7b39e4

## Thinking About Goal Scoring

### Shot Quality

If our project's aim is to somehow predict who is going to score goals, then it's worth looking at *how* goals are scored. Of course, each goal requires a shot to be taken, and for that shot to be on target, but the position on the pitch seems to have a material impact on the goal conversion rate.

(Note - we can infer the shot position from the text commentary).

**Share of shots on target, by position by shot position**

![ShotsOnTarget](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/ShotAccuracy.png)

**Share of shots converted to goals by shot position**

![GoalConversion](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/GoalConversion.png)

We can see that shots are not created equally. Generally, proximity to the goal is a good indicator as to whether a shot is going to be on target. However, we can see that goal conversion rates are wildly different - nearly 4 in 10 shots are goals if taken from very close proximity, compared to less than 3 in 100 if taken from a difficult angle.

As we would expect, penalties are frequently converted into goals, although these are pretty hard to account for in a model, since they are relatively rare events.

### Expected Goals

This analysis, differentiating shots by type and position, has another application.

Though the aim of this work is to predict goals, there are goals scored in the Premier League that just could not have been foreseen.

In gameweek 27, a central defender for Norwich, the bottom-placed team at that point, decided to have a punt from a very difficult angle against one of the league's best defences. It was a peach of a goal - but there's just no way any predictive model could have seen it coming. He could probably take a shot from the same position another ten times and not even come close to scoring.

![LewisGoal](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/Lewis.gif)

This illustrates the randomness that football contains as a sport, and why predicting goals is so tough. As well as cocky central defenders, goals can be caused by unlucky deflections, goalkeeping howlers, and freak movements of the ball. Similarly, nailed-on goals can be denied by unexpected goalie heroics, as well as the woodwork.

An alternative to trying to predict actual goals, therefore, is to work out how many goals we'd have expected a player to score, given the location and quality of the shots that they took.

Having defined the different types of possible shots (where on the pitch they were, whether they were struck with the foot or headers, whether they were directly assisted or not, etc.) we can see the goal conversion rate of each (expected goals, or XG). We can also look at the uplift in goal conversion if the shot is on target.

![ExpectedGoalTable](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/ShotTypes.png)

We can then look at the different shots that a player takes in a game, sum the 'XG' of each shot that they took, and create an overall 'Expected Goals' measure for that player in that game. Note - we can also do the same with assists (i.e. how many assists would we have expected the player to get based on the quality of the subsequent shots) and overall expected goal involvement (XGI, expected goals plus expected assists).

As it happens, XGI correlates pretty well with actual goal involvement.

![XGIViolins](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/XGIViolins.png)

XG and XGI also seem to increase in a linear fashion - i.e. the mean XG of players who scored twice in a match have roughly double the XG of players who only scored once.

![XGPlots](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/XGPlots.png)

### Home and Away
It's also worth noting the effect that playing at a team's home stadium, vs. other teams' stadiums can have on a team's ability to score (and indeed their ability to stop their opponents from scoring.

![HomeAwayGoals](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/HomeAwayGoals.png)

![HomeAwayGoalsCon](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/HomeAwayGoalsCon.png)

We should bear this in mind when we think about trying to guess how likely a player is to score in an upcoming game.

## Modelling

It's worth saying at the top of this section - I was not able to create high-quality predictive models using this dataset. As I'll discuss, the models were very bad at generalising, despite scoring well on the training dataset. Clearly, the random quirks of individual games caused the models to overfit, despite the attempts to mitigate randomness through the XG and XGI target variables.

I believe that it's possible to get better results than what I achieved with the data that was collected, though I expect that this would require further feature engineering, dimension reduction, (and no small amount of trial and error). Unfortunately, this has not been possible given time constraints for the project.

### Building The Dataset

The dataset created for the model training consisted of rows representing one player's performance in one game. In particular, it included features:

* The following stats for the last match, the average for the last 4 total matches, and the last 4 home/away matches as appropriate:
    * Shots and goals taken by type per minute
    * Shots and goals created by type per minute
    * The total minutes played
    * The relative strength of the opposition (given on a scale from -3 to +3)
* The following stats for the upcoming opposition for their last 4 total matches, and their last 4 home/away matches as appropriate:
    * Shots and goals conceded by type
    * Relative strength of their previous opponents
    * Other match stats such as clearances, ball touches, tackles made, etc.

### Training Predictive Models

The general approach to model training was the same for each type of algorithm:

* Create a train test split
* Train a vanilla model for benchmarking purposes
* Perform a cross validation grid search to tune hyperparameters USING THE TRAINING SET ONLY
* Test the tuned model on the test set

As mentioned, none of the models produced especially strong results when exposed to the test dataset.

![ModelResults](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/ModelResults.png)

This is shown when we visualise the model's predictions - this is taken from the Random Forest's attempt to predict the XG target:

![Predictions](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/Predictions.png)

We can see that the training set doesn't do too badly (the cross validation process during the gridsearch prevents it from overfitting too badly), but it really has no idea what to do with the test set. The model generally seems to suffer from bias - it is very very hesitant to predict any XGs above 1.

## Dashboard Building

Though we have been unable to build strong predictive models, we can still 'productise' the data we've gathered in a way that will help us with Fantasy Premier League.

In particular, we can use Tableau to create a series of interactive dashboards, allowing FPL managers to explore the various KPIs we've scraped and created. These have a number of different applications and use cases, which should help inform the strategy of FPL managers.

**Player Dashboard**

* A fully customisable dashboard showing KPIs of the user's choosing for each player
* Can filter performance based on player price, gameweek, and whether user wants to consider performance in home/away matches
* User selects players in scatter, which acts as a filter for the time series and rankings in the lower charts

![PlayerDashboard](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/PlayerDashboard.gif)

**Team Dashboard**

* A dashboard showing team-level performance across KPIs
* All views and KPIs are fully customisable and can be filtered as required by the user
* Gives a view of how teams have played against each other individual teams

![TeamDashboard](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/TeamDashboard.gif)

**Season Summaries**

* Shows how players have progressed through the season so far (by position), based on KPIs of the user's choice 
* Can be used to identify trends before they become obvious to every manager

![SeasonSummary](https://github.com/calbal91/project-moneyballing-fpl/blob/master/Images/SeasonSummary.gif)


These dashboards can be found at https://www.cb91.io/projects/fpl
