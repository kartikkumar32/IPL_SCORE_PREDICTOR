"""
utils/encoders.py
Label-encoding maps that mirror the sklearn LabelEncoder used during training.
All lists are sorted alphabetically – matching the default LabelEncoder order.
"""

# ── IPL Teams (alphabetical order = encoding index) ──────────────────────────
TEAMS = sorted([
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
])

TEAM_ENC: dict[str, int] = {team: idx for idx, team in enumerate(TEAMS)}

# ── IPL Venues (alphabetical order = encoding index) ─────────────────────────
VENUES = sorted([
    "Arun Jaitley Stadium",
    "Brabourne Stadium",
    "DY Patil Stadium",
    "Eden Gardens",
    "Feroz Shah Kotla",
    "MA Chidambaram Stadium",
    "MCA Stadium",
    "Maharashtra Cricket Association Stadium",
    "Narendra Modi Stadium",
    "Punjab Cricket Association Stadium",
    "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium",
    "Wankhede Stadium",
])

VENUE_ENC: dict[str, int] = {venue: idx for idx, venue in enumerate(VENUES)}

# ── Phase encoding ────────────────────────────────────────────────────────────
PHASE_LABELS = {
    0: "Powerplay (Ov 1–6)",
    1: "Middle Overs (Ov 7–15)",
    2: "Death Overs (Ov 16–20)",
}


def encode_team(name: str) -> int:
    """Return integer encoding for a team name. Defaults to 0 if not found."""
    return TEAM_ENC.get(name, 0)


def encode_venue(name: str) -> int:
    """Return integer encoding for a venue name. Defaults to 0 if not found."""
    return VENUE_ENC.get(name, 0)


def get_phase(over: int) -> int:
    """
    Returns match phase (0/1/2) based on over number.
      0 = Powerplay   (overs 1–6)
      1 = Middle overs(overs 7–15)
      2 = Death overs (overs 16–20)
    """
    if over < 6:
        return 0
    elif over < 15:
        return 1
    else:
        return 2
