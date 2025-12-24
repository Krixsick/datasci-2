"""
VCT 2025 Stats API + AI Insights
=================================

FastAPI backend with Gemini AI integration for Valorant esports analysis.

Run with: uv run uvicorn main:app --reload
API docs: http://localhost:8000/docs

"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel
from typing import Optional
import json

# =============================================================================
# APP SETUP
# =============================================================================
# load_dotenv()

app = FastAPI(
    title="VCT 2025 Stats API + AI",
    description="Valorant esports data analysis with Gemini AI insights",
    version="2.0.0"
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

MATCHES_PATH = Path("valorant/vct_2025/matches/overview.csv")
PLAYERS_PATH = Path("valorant/vct_2025/players_stats/players_stats.csv")

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

# =============================================================================
# GEMINI AI SETUP
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    AI_ENABLED = True
    print("✅ Gemini AI configured")
else:
    AI_ENABLED = False
    model = None
    print("⚠️ GEMINI_API_KEY not set. AI features disabled.")
    print("   Set it with: export GEMINI_API_KEY='your-key-here'")

# =============================================================================
# COLUMN CONFIGURATION - Update these to match your data!
# =============================================================================

PLAYER_COL = 'Player'
TEAM_COL = 'Teams'
AGENT_COL = 'Agents'
ACS_COL = 'Average Combat Score'
MAP_COL = 'Map'
KILLS_COL = 'Kills'
DEATHS_COL = 'Deaths'
ASSISTS_COL = 'Assists'
RATING_COL = 'Rating'

# =============================================================================
# PYDANTIC MODELS (for request/response validation)
# =============================================================================

class AIQueryRequest(BaseModel):
    """Request model for AI queries."""
    question: str
    
class AIResponse(BaseModel):
    """Response model for AI endpoints."""
    question: str
    answer: str
    data_used: Optional[dict] = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_player_stats_summary(player_name: str) -> dict:
    """Get a summary of player stats for AI context."""
    player_df = df_players[df_players[PLAYER_COL].str.lower() == player_name.lower()]
    
    if player_df.empty:
        return None
    
    # Calculate comprehensive stats
    stats = {
        "player": player_df[PLAYER_COL].iloc[0],
        "games_played": len(player_df),
    }
    
    # Add team
    if TEAM_COL in player_df.columns:
        stats["team"] = player_df[TEAM_COL].iloc[0]
    
    # Add numeric averages
    numeric_cols = player_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        stats[f"avg_{col.lower().replace(' ', '_')}"] = round(player_df[col].mean(), 2)
    
    # Add agent pool
    if AGENT_COL in player_df.columns:
        stats["agents_played"] = player_df[AGENT_COL].value_counts().head(5).to_dict()
    
    # Add performance trend (first half vs second half of games)
    if len(player_df) >= 4 and ACS_COL in player_df.columns:
        mid = len(player_df) // 2
        first_half_acs = player_df.head(mid)[ACS_COL].mean()
        second_half_acs = player_df.tail(mid)[ACS_COL].mean()
        stats["trend"] = "improving" if second_half_acs > first_half_acs else "declining"
    
    return stats


def get_top_players_data(n: int = 10) -> list:
    """Get top N players for AI context."""
    if ACS_COL not in df_players.columns:
        return []
    
    player_stats = df_players.groupby(PLAYER_COL).agg({
        ACS_COL: 'mean',
        PLAYER_COL: 'count'
    }).rename(columns={PLAYER_COL: 'games', ACS_COL: 'avg_acs'})
    
    qualified = player_stats[player_stats['games'] >= 5]
    top = qualified.nlargest(n, 'avg_acs').reset_index()
    
    return top.to_dict(orient='records')


def get_agent_meta() -> list:
    """Get agent pick rates and performance for AI context."""
    if AGENT_COL not in df_players.columns:
        return []
    
    agent_stats = df_players.groupby(AGENT_COL).agg({
        ACS_COL: 'mean' if ACS_COL in df_players.columns else 'count',
        PLAYER_COL: 'count'
    })
    
    if ACS_COL in df_players.columns:
        agent_stats.columns = ['avg_acs', 'times_played']
        agent_stats['avg_acs'] = agent_stats['avg_acs'].round(1)
    else:
        agent_stats.columns = ['times_played', 'count']
    
    return agent_stats.reset_index().to_dict(orient='records')


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    """API status and available endpoints."""
    return {
        "status": "running",
        "data_loaded": DATA_LOADED,
        "ai_enabled": AI_ENABLED,
        "records": {
            "matches": len(df_matches),
            "player_stats": len(df_players)
        },
        "endpoints": {
            "data": {
                "players_top": "/api/players/top",
                "player_profile": "/api/player/{name}",
                "agents": "/api/agents",
                "teams": "/api/teams",
                "maps": "/api/maps"
            },
            "ai": {
                "player_analysis": "/api/ai/player/{name}",
                "ask": "/api/ai/ask",
                "team_comp": "/api/ai/team-comp",
                "match_prediction": "/api/ai/match-prediction"
            }
        },
        "docs": "/docs"
    }


@app.get("/api/columns")
def get_columns():
    """Debug: See available columns."""
    return {
        "matches_columns": df_matches.columns.tolist() if DATA_LOADED else [],
        "players_columns": df_players.columns.tolist() if DATA_LOADED else [],
    }


# =============================================================================
# DATA ENDPOINTS (Original functionality)
# =============================================================================

@app.get("/api/players/top")
def get_top_players(
    min_games: int = Query(default=5, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    sort_by: str = Query(default="avg_acs")
):
    """Get top players by ACS."""
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
    
    qualified = player_stats[player_stats['games'] >= min_games]
    
    if sort_by == 'games':
        result = qualified.sort_values('games', ascending=False)
    else:
        result = qualified.sort_values('avg_acs', ascending=False)
    
    return {
        "total_qualified": len(qualified),
        "players": result.head(limit).reset_index().to_dict(orient="records")
    }


@app.get("/api/player/{player_name}")
def get_player_profile(player_name: str):
    """Get detailed stats for a player."""
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    player_df = df_players[df_players[PLAYER_COL].str.lower() == player_name.lower()]
    
    if player_df.empty:
        partial = df_players[df_players[PLAYER_COL].str.lower().str.contains(player_name.lower(), na=False)]
        suggestions = partial[PLAYER_COL].unique().tolist()[:5]
        raise HTTPException(status_code=404, detail=f"Player not found. Try: {suggestions}")
    
    numeric_cols = player_df.select_dtypes(include=['int64', 'float64']).columns
    avg_stats = {col: round(player_df[col].mean(), 2) for col in numeric_cols}
    
    result = {
        "player": player_df[PLAYER_COL].iloc[0],
        "games_played": len(player_df),
        "averages": avg_stats
    }
    
    if TEAM_COL in player_df.columns:
        result["team"] = player_df[TEAM_COL].iloc[0]
    
    if AGENT_COL in player_df.columns:
        result["agents"] = player_df[AGENT_COL].value_counts().to_dict()
    
    return result


@app.get("/api/players/search")
def search_players(q: str = Query(..., min_length=2)):
    """Search players by name."""
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    mask = df_players[PLAYER_COL].str.lower().str.contains(q.lower(), na=False)
    matches = df_players[mask][PLAYER_COL].unique().tolist()
    
    return {"query": q, "results": matches[:20], "total": len(matches)}


@app.get("/api/agents")
def get_agent_stats():
    """Get performance by agent."""
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if AGENT_COL not in df_players.columns:
        return {"error": f"Agent column '{AGENT_COL}' not found"}
    
    agent_stats = df_players.groupby(AGENT_COL).agg({
        ACS_COL: 'mean' if ACS_COL in df_players.columns else 'count',
        PLAYER_COL: 'count'
    })
    
    agent_stats.columns = ['avg_acs', 'times_played']
    agent_stats['avg_acs'] = agent_stats['avg_acs'].round(1)
    agent_stats = agent_stats.sort_values('avg_acs', ascending=False)
    
    return {"agents": agent_stats.reset_index().to_dict(orient="records")}


@app.get("/api/teams")
def get_team_stats(min_games: int = Query(default=20)):
    """Get performance by team."""
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
    
    qualified = team_stats[team_stats['total_games'] >= min_games]
    qualified = qualified.sort_values('avg_acs', ascending=False)
    
    return {"teams": qualified.reset_index().to_dict(orient="records")}


@app.get("/api/maps")
def get_map_stats():
    """Get map play counts."""
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if MAP_COL not in df_matches.columns:
        return {"error": f"Map column '{MAP_COL}' not found"}
    
    map_counts = df_matches[MAP_COL].value_counts()
    
    return {"maps": [{"map": name, "times_played": int(count)} for name, count in map_counts.items()]}


@app.get("/api/compare")
def compare_players(players: str = Query(..., description="Comma-separated names")):
    """Compare multiple players."""
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


# =============================================================================
# AI-POWERED ENDPOINTS (New!)
# =============================================================================

@app.get("/api/ai/player/{player_name}")
def ai_player_analysis(player_name: str):
    """
    🤖 AI-Generated Player Analysis
    
    Uses Gemini to create a detailed scouting report for any player.
    """
    if not AI_ENABLED:
        raise HTTPException(status_code=503, detail="AI not configured. Set GEMINI_API_KEY env variable.")
    
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Get player stats
    stats = get_player_stats_summary(player_name)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    
    # Create prompt for Gemini
    prompt = f"""You are an expert Valorant esports analyst. Based on the following player statistics, 
write a brief scouting report (3-4 paragraphs). Include:
1. Overall assessment of their skill level
2. Their role and agent preferences
3. Strengths and potential weaknesses
4. How they compare to typical pro players

Player Stats:
{json.dumps(stats, indent=2)}

Context: Average pro ACS is around 200-220. Elite players average 250+.
Keep the analysis concise, insightful, and data-driven."""

    try:
        response = model.generate_content(prompt)
        
        return {
            "player": stats["player"],
            "analysis": response.text,
            "stats_used": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@app.post("/api/ai/ask")
def ai_ask_question(request: AIQueryRequest):
    """
    🤖 Ask Questions About VCT Data
    """
    if not AI_ENABLED:
        raise HTTPException(status_code=503, detail="AI not configured. Set GEMINI_API_KEY env variable.")
    
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        # Gather context data for the AI
        context = {
            "top_players": get_top_players_data(15),
            "agent_meta": get_agent_meta(),
            "total_players": df_players[PLAYER_COL].nunique() if PLAYER_COL in df_players.columns else 0,
            "total_matches": len(df_matches)
        }
        
        print("DEBUG: Context gathered successfully")  # Debug line
        
        prompt = f"""You are a Valorant esports data analyst. Answer the following question using ONLY the data provided.
If the data doesn't contain enough information to answer, say so.

Question: {request.question}

Available Data:
- Top 15 Players by ACS: {json.dumps(context['top_players'], indent=2)}
- Agent Statistics: {json.dumps(context['agent_meta'], indent=2)}
- Total unique players in dataset: {context['total_players']}
- Total matches in dataset: {context['total_matches']}

Provide a clear, concise answer based on this data."""

        print("DEBUG: Calling Gemini...")  # Debug line
        response = model.generate_content(prompt)
        print("DEBUG: Gemini responded successfully")  # Debug line
        
        return {
            "question": request.question,
            "answer": response.text,
            "data_used": context
        }
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())  # This prints the full error to terminal
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

@app.get("/api/ai/team-comp")
def ai_team_composition(
    role: str = Query(default="duelist", description="Role: duelist, controller, initiator, sentinel"),
    map_name: str = Query(default=None, description="Specific map (optional)")
):
    """
    🤖 AI Team Composition Advisor
    
    Get AI-powered recommendations for team compositions.
    """
    if not AI_ENABLED:
        raise HTTPException(status_code=503, detail="AI not configured. Set GEMINI_API_KEY env variable.")
    
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Get agent performance data
    agent_data = get_agent_meta()
    
    # Get top players for each agent
    top_agent_players = {}
    if AGENT_COL in df_players.columns:
        for agent_info in agent_data[:10]:  # Top 10 agents
            agent_name = agent_info.get(AGENT_COL)
            if agent_name:
                agent_df = df_players[df_players[AGENT_COL] == agent_name]
                if not agent_df.empty and ACS_COL in agent_df.columns:
                    top_player = agent_df.groupby(PLAYER_COL)[ACS_COL].mean().nlargest(1)
                    if not top_player.empty:
                        top_agent_players[agent_name] = {
                            "best_player": top_player.index[0],
                            "avg_acs": round(top_player.values[0], 1)
                        }
    
    prompt = f"""You are a Valorant coach and analyst. Based on the current meta data, suggest a team composition.

Role requested: {role}
Map: {map_name if map_name else "General/Any map"}

Agent Performance Data (pick rate and avg ACS):
{json.dumps(agent_data, indent=2)}

Top Players per Agent:
{json.dumps(top_agent_players, indent=2)}

Please provide:
1. Recommended agent for the {role} role with reasoning
2. A full 5-agent team composition suggestion
3. Brief explanation of why this comp works
4. Name a pro player who excels at each role if data is available

Keep the response concise and actionable."""

    try:
        response = model.generate_content(prompt)
        
        return {
            "role_requested": role,
            "map": map_name or "any",
            "recommendation": response.text,
            "agent_data": agent_data[:10],
            "top_players_by_agent": top_agent_players
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@app.get("/api/ai/match-prediction")
def ai_match_prediction(
    team1: str = Query(..., description="First team name"),
    team2: str = Query(..., description="Second team name")
):
    """
    🤖 AI Match Prediction
    
    Get AI analysis of a potential matchup between two teams.
    """
    if not AI_ENABLED:
        raise HTTPException(status_code=503, detail="AI not configured. Set GEMINI_API_KEY env variable.")
    
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Get stats for both teams
    def get_team_data(team_name: str) -> dict:
        team_df = df_players[df_players[TEAM_COL].str.lower().str.contains(team_name.lower(), na=False)]
        
        if team_df.empty:
            return None
        
        return {
            "team": team_df[TEAM_COL].iloc[0],
            "players": team_df[PLAYER_COL].unique().tolist()[:5],
            "avg_acs": round(team_df[ACS_COL].mean(), 1) if ACS_COL in team_df.columns else None,
            "games_in_data": len(team_df),
            "top_performer": team_df.groupby(PLAYER_COL)[ACS_COL].mean().idxmax() if ACS_COL in team_df.columns else None
        }
    
    team1_data = get_team_data(team1)
    team2_data = get_team_data(team2)
    
    if not team1_data:
        raise HTTPException(status_code=404, detail=f"Team '{team1}' not found")
    if not team2_data:
        raise HTTPException(status_code=404, detail=f"Team '{team2}' not found")
    
    prompt = f"""You are an esports analyst predicting a Valorant match. Analyze this matchup:

Team 1: {json.dumps(team1_data, indent=2)}

Team 2: {json.dumps(team2_data, indent=2)}

Provide:
1. Which team has the statistical advantage and why
2. Key players to watch from each side
3. What each team needs to do to win
4. Your prediction with confidence level (based only on the stats provided)

Be balanced and acknowledge the limitations of predicting from stats alone."""

    try:
        response = model.generate_content(prompt)
        
        return {
            "team1": team1_data,
            "team2": team2_data,
            "analysis": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")