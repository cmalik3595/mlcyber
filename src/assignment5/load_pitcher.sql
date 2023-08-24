-- Create a lookup table for all pitchers among all games.
DROP TEMPORARY TABLE IF EXISTS t_pitcher_lookup;
CREATE TEMPORARY TABLE t_pitcher_lookup engine=MyISAM AS
SELECT
	g.game_id,
	g.local_date,
	pc.pitcher,
	pc.endingInning - pc.startingInning + 1 AS innings_played,
	pc.Strikeout AS strikeouts,
	pc.Hit AS hits,
	pc.Home_run AS home_runs
FROM pitcher_counts pc
JOIN game g On g.game_id = pc.game_id
ORDER BY g.game_id;

CREATE UNIQUE INDEX pl_game_date_pitcher_idx
	ON t_pitcher_lookup (game_id, local_date, pitcher);
CREATE UNIQUE INDEX pl_game_pitcher_idx
	ON t_pitcher_lookup (game_id, pitcher);
CREATE UNIQUE INDEX pl_date_pitcher_idx
	ON t_pitcher_lookup (local_date, pitcher);
CREATE INDEX pl_game_idx ON t_pitcher_lookup (game_id);
CREATE INDEX pl_date_idx ON t_pitcher_lookup (local_date);
CREATE INDEX pl_pitcher_idx ON t_pitcher_lookup (pitcher);


-- -- -- -- -- -- -- -- -- 
-- 100 DAY ROLLING STATS --
-- -- -- -- -- -- -- -- -- 

-- Calculate 100 day rolling stats for all pitchers that had started a game.
DROP TEMPORARY TABLE IF EXISTS  t_pitcher_rolling_100;
CREATE TEMPORARY TABLE t_pitcher_rolling_100 engine=MyISAM AS
SELECT
	l.pitcher,
	l.game_id,
	l.local_date,
	SUM(r.innings_played) AS innings_played,
	COUNT(DISTINCT(r.game_id)) AS games_started,
	9*(SUM(r.strikeouts)/SUM(r.innings_played)) AS strikeouts_9,
	9*(SUM(r.hits)/SUM(r.innings_played)) AS hits_9,
	9*(SUM(r.home_runs)/SUM(r.innings_played)) AS home_runs_9
FROM t_pitcher_lookup l
LEFT JOIN t_pitcher_lookup r ON l.pitcher = r.pitcher
	AND r.local_date 
	BETWEEN DATE_SUB(l.local_date, INTERVAL 100 DAY) AND l.local_date - INTERVAL 1 DAY
GROUP BY
	l.pitcher,
	l.game_id, 
	l.local_date
ORDER BY game_id;

-- Create a temp table for every game's home starting pitcher stats
DROP TEMPORARY TABLE IF EXISTS t_home_starting_pitcher_stats_100;
CREATE TEMPORARY TABLE t_home_starting_pitcher_stats_100 engine=MyISAM AS
SELECT
	game_id,
	local_date,
	pitcher,
	innings_played,
	games_started,
	strikeouts_9,
	hits_9,
	home_runs_9
FROM (
	SELECT
		pr.*, pc.homeTeam, pc.startingPitcher
	FROM t_pitcher_rolling_100 pr
	JOIN pitcher_counts pc ON pr.game_id = pc.game_id 
		AND pr.pitcher = pc.pitcher
	ORDER BY game_id
) AS prpc
WHERE homeTeam = 1 AND startingPitcher = 1;
CREATE INDEX hsp100_game_idx
	ON t_home_starting_pitcher_stats_100 (game_id);


-- Create a temp table for every game's home starting pitcher stats
DROP TEMPORARY TABLE IF EXISTS t_away_starting_pitcher_stats_100;
CREATE TEMPORARY TABLE t_away_starting_pitcher_stats_100 engine=MyISAM AS
SELECT
	game_id,
	local_date,
	pitcher,
	innings_played,
	games_started,
	strikeouts_9,
	hits_9,
	home_runs_9
FROM (
	SELECT
		pr.*, pc.awayTeam, pc.startingPitcher
	FROM t_pitcher_rolling_100 pr
	JOIN pitcher_counts pc ON pr.game_id = pc.game_id 
		AND pr.pitcher = pc.pitcher
	ORDER BY game_id
) AS prpc
WHERE awayTeam = 1 AND startingPitcher = 1;
CREATE INDEX asp100_game_idx
	ON t_away_starting_pitcher_stats_100 (game_id);

-- Create a permanant combined table for home and relative stats.
DROP TABLE IF EXISTS starting_pitcher_stats_100;
CREATE TABLE starting_pitcher_stats_100 engine=MyISAM AS
SELECT
	h.game_id,
	h.local_date,
	h.pitcher AS home_pitcher,
	a.pitcher AS away_pitcher,
	h.innings_played AS ip,
	h.games_started AS gs,
	h.strikeouts_9 AS s9,
	h.hits_9 AS h9,
	h.home_runs_9 AS hr9,
	h.innings_played/a.innings_played AS ip_ratio,
	h.games_started-a.games_started AS gs_difference,
	h.strikeouts_9-a.strikeouts_9 AS s9_difference,
	h.hits_9-a.hits_9 AS h9_difference,
	h.home_runs_9-a.home_runs_9 AS hr9_difference
FROM t_home_starting_pitcher_stats_100 h
JOIN t_away_starting_pitcher_stats_100 a ON h.game_id = a.game_id;

-- -- -- -- -- -- -- -- -- 
-- CAREER STATS --
-- -- -- -- -- -- -- -- -- 

-- Calculate prior career stats for all pitchers that had started a game.
DROP TEMPORARY TABLE IF EXISTS  t_pitcher_prior_career;
CREATE TEMPORARY TABLE t_pitcher_rolling_prior_career engine=MyISAM AS
SELECT
	l.pitcher,
	l.game_id,
	l.local_date,
	SUM(r.innings_played) AS innings_played,
	COUNT(DISTINCT(r.game_id)) AS games_started,
	9*(SUM(r.strikeouts)/SUM(r.innings_played)) AS strikeouts_9,
	9*(SUM(r.hits)/SUM(r.innings_played)) AS hits_9,
	9*(SUM(r.home_runs)/SUM(r.innings_played)) AS home_runs_9
FROM t_pitcher_lookup l
LEFT JOIN t_pitcher_lookup r ON l.pitcher = r.pitcher
	AND r.local_date < l.local_date
GROUP BY
	l.pitcher,
	l.game_id, 
	l.local_date
ORDER BY game_id;

DROP TEMPORARY TABLE IF EXISTS t_pitcher_prior_career;

-- Create a temp table for every game's home starting pitcher stats
DROP TEMPORARY TABLE IF EXISTS t_home_starting_pitcher_stats_prior_career;
CREATE TEMPORARY TABLE t_home_starting_pitcher_stats_prior_career engine=MyISAM AS
SELECT
	game_id,
	local_date,
	pitcher,
	innings_played,
	games_started,
	strikeouts_9,
	hits_9,
	home_runs_9
FROM (
	SELECT
		pr.*, pc.homeTeam, pc.startingPitcher
	FROM t_pitcher_rolling_prior_career pr
	JOIN pitcher_counts pc ON pr.game_id = pc.game_id 
		AND pr.pitcher = pc.pitcher
	ORDER BY game_id
) AS prpc
WHERE homeTeam = 1 AND startingPitcher = 1;
CREATE INDEX hsppc_game_idx
	ON t_home_starting_pitcher_stats_prior_career (game_id);


-- Create a temp table for every game's home starting pitcher stats
DROP TEMPORARY TABLE IF EXISTS t_away_starting_pitcher_stats_prior_career;
CREATE TEMPORARY TABLE t_away_starting_pitcher_stats_prior_career engine=MyISAM AS
SELECT
	game_id,
	local_date,
	pitcher,
	innings_played,
	games_started,
	strikeouts_9,
	hits_9,
	home_runs_9
FROM (
	SELECT
		pr.*, pc.awayTeam, pc.startingPitcher
	FROM t_pitcher_rolling_prior_career pr
	JOIN pitcher_counts pc ON pr.game_id = pc.game_id 
		AND pr.pitcher = pc.pitcher
	ORDER BY game_id
) AS prpc
WHERE awayTeam = 1 AND startingPitcher = 1;
CREATE INDEX asppc_game_idx
	ON t_away_starting_pitcher_stats_prior_career (game_id);

-- Create a permanant combined table for home and relative stats.
DROP TABLE IF EXISTS starting_pitcher_stats_prior_career;
CREATE TABLE starting_pitcher_stats_prior_career engine=MyISAM AS
SELECT
	h.game_id,
	h.local_date,
	h.pitcher AS home_pitcher,
	a.pitcher AS away_pitcher,
	h.innings_played AS ip,
	h.games_started AS gs,
	h.strikeouts_9 AS s9,
	h.hits_9 AS h9,
	h.home_runs_9 AS hr9,
	h.innings_played/a.innings_played AS ip_ratio,
	h.games_started-a.games_started AS gs_difference,
	h.strikeouts_9-a.strikeouts_9 AS s9_difference,
	h.hits_9-a.hits_9 AS h9_difference,
	h.home_runs_9-a.home_runs_9 AS hr9_difference
FROM t_home_starting_pitcher_stats_prior_career h
JOIN t_away_starting_pitcher_stats_prior_career a ON h.game_id = a.game_id
