# Data sources and licensing

Checked on 5 October 2026.

## ATP match results

- **Original producer:** Jeff Sackmann / Tennis Abstract.
- **Original repository:** <https://github.com/JeffSackmann/tennis_atp>
- **Current distribution used:**
  <https://github.com/Aneeshers/tennis-sackmann-archive/tree/main/atp>
- **Status:** the original repository returned 404 during project setup. The
  archival mirror states that its ATP snapshot was taken in June 2026 and
  contains season files through 2026.
- **License asserted by the archive:** Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Commercial use:** **No**, unless separate permission is obtained from the
  relevant rights holder. Attribution and ShareAlike are also required.
- **Format:** one winner/loser-oriented match CSV per season plus a player
  reference CSV. Match rows include tournament metadata, rankings, age/height,
  score, duration and service statistics when recorded.

The pipeline does not silently combine this source with another match-results
provider. It converts winner/loser source columns into outcome-independent A/B
participant slots in the processed layer while preserving provenance.

## Grand Slam point-by-point

- **Original producer/repository:** Jeff Sackmann,
  <https://github.com/JeffSackmann/tennis_slam_pointbypoint>
- **Current distribution used:**
  <https://github.com/Aneeshers/tennis-sackmann-archive/tree/main/slam_pointbypoint>
- **Snapshot/coverage:** upstream commit `6febb77` (October 2024), global range
  2011–2024. Coverage is not complete for every event/year: Wimbledon 2020
  (cancelled) and Australian Open/Roland-Garros 2022–2024 are absent.
- **License asserted by the archive:** CC BY-NC-SA 4.0.
- **Commercial use:** **No** without separate permission.
- **Format:** a `*-matches.csv` metadata file and `*-points.csv` event file per
  tournament/year. Point fields include server, point winner and score state;
  coverage and encoding differ from match-level ATP files.

The point schema itself is not assumed constant across years/providers. For
example, the 2024 US Open archive exposes `PointServer`, `PointWinner`, serve
speed/direction, rally count and several derived error/winner indicators,
whereas other files can have fewer or differently named fields. The first phase
keeps each original CSV intact and does not concatenate heterogeneous schemas.

This dataset stays under `data/external/slam_pointbypoint` and is not joined to
ATP matches in the first pipeline version.

## Match Charting Project (MCP)

- **Producer/repository:** Tennis Abstract contributors / Jeff Sackmann,
  <https://github.com/JeffSackmann/tennis_MatchChartingProject>
- **Status:** accessible and active when checked.
- **License:** CC BY-NC-SA 4.0, explicitly stated by the project.
- **Commercial use:** **No** without separate permission.
- **Format:** match metadata plus point rows. `Player1` is the player who served
  first, not the winner. The `1st` and `2nd` strings are contributed charting
  codes for serve direction, shot type/direction/depth and point outcome;
  derived columns expose ace, double fault, rally winner/error and rally count.

The current men's point files are partitioned into `to-2009`, `2010s` and
`2020s`; their core columns include match ID, point sequence, set/game/point
score, server, first-serve code, second-serve code, notes and point winner.

MCP is shot-level and crowdsourced; Grand Slam point-by-point is event-level.
They are separate datasets with separate directories and manifest entries.

## Provenance guarantees

Every downloaded artifact receives a `data/manifest.json` record containing its
URL, original source URL, distribution URL, retrieval time, byte size, SHA-256,
year, source snapshot note and license declaration. Existing files are verified
and reused; the pipeline never replaces one in place.

The manifest records what was fetched, but it is not legal advice. Before any
commercial product, obtain suitable permission or replace these sources with a
commercially licensed feed.
