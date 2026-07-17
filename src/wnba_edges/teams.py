from __future__ import annotations

TEAM_NAME_TO_ABBR = {
    "Atlanta Dream": "ATL",
    "Chicago Sky": "CHI",
    "Connecticut Sun": "CON",
    "Dallas Wings": "DAL",
    "Golden State Valkyries": "GSV",
    "Indiana Fever": "IND",
    "Las Vegas Aces": "LVA",
    "Los Angeles Sparks": "LAS",
    "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL",
    "Phoenix Mercury": "PHX",
    "Portland Fire": "PDX",
    "Seattle Storm": "SEA",
    "Toronto Tempo": "TOR",
    "Washington Mystics": "WAS",
}

KNOWN_ABBRS = set(TEAM_NAME_TO_ABBR.values())

# Other sites use their own short codes (ESPN etc.); normalize to ours.
ALT_ABBR_TO_ABBR = {
    "CONN": "CON",
    "GS": "GSV",
    "GSW": "GSV",
    "LA": "LAS",
    "LV": "LVA",
    "LVA": "LVA",
    "NY": "NYL",
    "PHO": "PHX",
    "POR": "PDX",
    "WSH": "WAS",
}


def team_abbr(name: str) -> str:
    """Normalize a team name or foreign abbreviation to the internal 3-letter code."""
    n = str(name).strip()
    if n.upper() in KNOWN_ABBRS:
        return n.upper()
    if n in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[n]
    if n.upper() in ALT_ABBR_TO_ABBR:
        return ALT_ABBR_TO_ABBR[n.upper()]
    return n.upper()[:3]


def is_known_team(name: str) -> bool:
    return team_abbr(name) in KNOWN_ABBRS
