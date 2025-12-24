"""
VCT 2025 Stats API
==================

FastAPI backend for Valorant esports data analysis.

Run with: uv run uvicorn main:app --reload
API docs: http://localhost:8000/docs

SETUP:
1. Run notebooks/01_exploration.ipynb to find your column names
2. Update the COLUMN constants below
3. Start the server!
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="VCT 2025 Stats API",
    description="Valorant esports data analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA LOADING
# =============================================================================

# Paths to your data
MATCHES_PATH = Path("valorant/vct_2025/matches/overview.csv")
PLAYERS_PATH = Path("valorant/vct_2025/players_stats/players_stats.csv")

# Load data at startup
try:
    df_matches = pd.read_csv(MATCHES_PATH)
    df_players = pd.read_csv(PLAYERS_PATH)
    DATA_LOADED = True
    print(f"✅ Loaded {len(df_matches):,} matches and {len(df_players):,} player records")
except FileNotFoundError as e:
    DATA_LOADED = False
    df_matches = pd.DataFrame()
    df_players = pd.DataFrame()
    print(f"⚠️ Data files not found: {e}")
    print("   Make sure you're running from the 'server' folder")

# =============================================================================
# COLUMN CONFIGURATION
# TODO: Update these after running 01_exploration.ipynb!
# =============================================================================

PLAYER_COL = 'Player'   # Column containing player names
TEAM_COL = 'Teams'       # Column containing team names
AGENT_COL = 'Agents'     # Column containing agent names
ACS_COL = 'Average Combat Score'         # Column containing ACS values
MAP_COL = 'Map'         # Column containing map names

# Add more as needed based on your data:
# KILLS_COL = 'K'
# DEATHS_COL = 'D'
# ASSISTS_COL = 'A'
# RATING_COL = 'Rating'


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/")
def root():
    """API status and available endpoints."""
    return {
        "status": "running",
        "data_loaded": DATA_LOADED,
        "records": {
            "matches": len(df_matches),
            "player_stats": len(df_players)
        },
        "endpoints": {
            "players": {
                "top": "/api/players/top - Top players by ACS",
                "search": "/api/players/search?q=name - Search players",
                "profile": "/api/player/{name} - Player profile"
            },
            "agents": "/api/agents - Agent statistics",
            "teams": "/api/teams - Team statistics",
            "maps": "/api/maps - Map statistics"
        },
        "docs": "/docs"
    }


@app.get("/api/columns")
def get_columns():
    """Debug endpoint: See all available columns."""
    return {
        "matches_columns": df_matches.columns.tolist() if DATA_LOADED else [],
        "players_columns": df_players.columns.tolist() if DATA_LOADED else [],
        "configured": {
            "PLAYER_COL": PLAYER_COL,
            "TEAM_COL": TEAM_COL,
            "AGENT_COL": AGENT_COL,
            "ACS_COL": ACS_COL,
            "MAP_COL": MAP_COL
        }
    }


# =============================================================================
# PLAYER ENDPOINTS
# =============================================================================

@app.get("/api/players/top")
def get_top_players(
    min_games: int = Query(default=5, ge=1, description="Minimum games played"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of players to return"),
    sort_by: str = Query(default="avg_acs", description="Sort by: avg_acs, games")
):
    """
    Get top players by average ACS.
    
    - **min_games**: Filter out players with fewer games
    - **limit**: How many players to return
    - **sort_by**: Column to sort by
    """
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if PLAYER_COL not in df_players.columns or ACS_COL not in df_players.columns:
        return {"error": "Column configuration incorrect. Check /api/columns"}
    
    player_stats = df_players.groupby(PLAYER_COL).agg({
        ACS_COL: 'mean',
        TEAM_COL: 'first' if TEAM_COL in df_players.columns else lambda x: 'Unknown',
        PLAYER_COL: 'count'
    })
    
    player_stats.columns = ['avg_acs', 'team', 'games']
    player_stats['avg_acs'] = player_stats['avg_acs'].round(1)
    
    # Filter and sort
    qualified = player_stats[player_stats['games'] >= min_games]
    
    if sort_by == 'games':
        result = qualified.sort_values('games', ascending=False)
    else:
        result = qualified.sort_values('avg_acs', ascending=False)
    
    return {
        "filters": {"min_games": min_games, "sort_by": sort_by},
        "total_qualified": len(qualified),
        "players": result.head(limit).reset_index().to_dict(orient="records")
    }


@app.get("/api/players/search")
def search_players(
    q: str = Query(..., min_length=2, description="Search query")
):
    """
    Search for players by name.
    """
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Case-insensitive search
    mask = df_players[PLAYER_COL].str.lower().str.contains(q.lower(), na=False)
    matches = df_players[mask][PLAYER_COL].unique().tolist()
    
    return {
        "query": q,
        "results": matches[:20],  # Limit to 20 results
        "total": len(matches)
    }


@app.get("/api/player/{player_name}")
def get_player_profile(player_name: str):
    """
    Get detailed stats for a specific player.
    """
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Case-insensitive match
    player_df = df_players[df_players[PLAYER_COL].str.lower() == player_name.lower()]
    
    if player_df.empty:
        # Try partial match
        partial = df_players[df_players[PLAYER_COL].str.lower().str.contains(player_name.lower(), na=False)]
        suggestions = partial[PLAYER_COL].unique().tolist()[:5]
        raise HTTPException(
            status_code=404, 
            detail=f"Player '{player_name}' not found. Did you mean: {suggestions}"
        )
    
    # Calculate stats
    numeric_cols = player_df.select_dtypes(include=['int64', 'float64']).columns
    avg_stats = {col: round(player_df[col].mean(), 2) for col in numeric_cols}
    
    result = {
        "player": player_df[PLAYER_COL].iloc[0],
        "games_played": len(player_df),
        "averages": avg_stats
    }
    
    # Add team if available
    if TEAM_COL in player_df.columns:
        result["team"] = player_df[TEAM_COL].iloc[0]
    
    # Add agent pool if available
    if AGENT_COL in player_df.columns:
        result["agents"] = player_df[AGENT_COL].value_counts().to_dict()
    
    return result


# =============================================================================
# AGENT ENDPOINTS
# =============================================================================

@app.get("/api/agents")
def get_agent_stats():
    """
    Get performance statistics by agent.
    """
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if AGENT_COL not in df_players.columns:
        return {"error": f"Agent column '{AGENT_COL}' not found"}
    
    agent_stats = df_players.groupby(AGENT_COL).agg({
        ACS_COL: 'mean' if ACS_COL in df_players.columns else 'count',
        PLAYER_COL: 'count'
    })
    
    if ACS_COL in df_players.columns:
        agent_stats.columns = ['avg_acs', 'times_played']
        agent_stats['avg_acs'] = agent_stats['avg_acs'].round(1)
        agent_stats = agent_stats.sort_values('avg_acs', ascending=False)
    else:
        agent_stats.columns = ['times_played', 'count']
        agent_stats = agent_stats.sort_values('times_played', ascending=False)
    
    return {
        "total_agents": len(agent_stats),
        "agents": agent_stats.reset_index().to_dict(orient="records")
    }


# =============================================================================
# TEAM ENDPOINTS
# =============================================================================

@app.get("/api/teams")
def get_team_stats(
    min_games: int = Query(default=20, description="Minimum total player-games")
):
    """
    Get performance statistics by team.
    """
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if TEAM_COL not in df_players.columns:
        return {"error": f"Team column '{TEAM_COL}' not found"}
    
    team_stats = df_players.groupby(TEAM_COL).agg({
        ACS_COL: 'mean' if ACS_COL in df_players.columns else 'count',
        PLAYER_COL: ['count', 'nunique']
    })
    
    team_stats.columns = ['avg_acs', 'total_games', 'unique_players']
    team_stats['avg_acs'] = team_stats['avg_acs'].round(1)
    
    # Filter by minimum games
    qualified = team_stats[team_stats['total_games'] >= min_games]
    qualified = qualified.sort_values('avg_acs', ascending=False)
    
    return {
        "filters": {"min_games": min_games},
        "total_teams": len(qualified),
        "teams": qualified.reset_index().to_dict(orient="records")
    }


# =============================================================================
# MAP ENDPOINTS
# =============================================================================

@app.get("/api/maps")
def get_map_stats():
    """
    Get statistics by map.
    """
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if MAP_COL not in df_matches.columns:
        return {"error": f"Map column '{MAP_COL}' not found in matches data"}
    
    map_counts = df_matches[MAP_COL].value_counts()
    
    return {
        "total_maps": len(map_counts),
        "maps": [{"map": name, "times_played": int(count)} for name, count in map_counts.items()]
    }


# =============================================================================
# COMPARISON ENDPOINTS
# =============================================================================

@app.get("/api/compare")
def compare_players(
    players: str = Query(..., description="Comma-separated player names")
):
    """
    Compare multiple players side by side.
    
    Example: /api/compare?players=TenZ,yay,aspas
    """
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    player_list = [p.strip() for p in players.split(',')]
    
    results = []
    for name in player_list:
        player_df = df_players[df_players[PLAYER_COL].str.lower() == name.lower()]
        
        if not player_df.empty:
            stats = {
                "player": player_df[PLAYER_COL].iloc[0],
                "games": len(player_df),
                "avg_acs": round(player_df[ACS_COL].mean(), 1) if ACS_COL in player_df.columns else None
            }
            if TEAM_COL in player_df.columns:
                stats["team"] = player_df[TEAM_COL].iloc[0]
            results.append(stats)
        else:
            results.append({"player": name, "error": "not found"})
    
    return {"comparison": results}