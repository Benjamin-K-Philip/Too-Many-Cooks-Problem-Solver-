"""
 The 2 "too many cooks" problems used in the Build T:

  1. MERGE  - Five people edited the same document. Conflicting rewrites,
              disputed deletions, duplicate typo fixes. The tool resolves
              every conflict with voting + similarity clustering + tie-breaks
              and produces ONE consistent final draft, with a decision log.

  2. ASSIGN - A football team where all 11 players want to be the striker.
              The tool uses the Hungarian algorithm (optimal assignment)
              to give each player the role that maximizes TOTAL team skill,
              and compares it against the "everyone plays striker" chaos.

Usage:
    python too_many_cooks.py merge  edits.json      # resolve document edits
    python too_many_cooks.py assign players.json    # assign team roles
    python too_many_cooks.py demo                   # run both with sample data

Dependencies used: Python standard library + scipy (for the assignment solver).
"""

import json
import sys
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
from scipy.optimize import linear_sum_assignment


# ======================================================================
# PROBLEM 1: THE DOCUMENT MERGER
# ----------------------------------------------------------------------
# Decision logic used (in order):
#   a) DELETE vs KEEP  -> majority vote among editors who touched the
#      sentence. A paragraph one person polished can't be silently
#      deleted by one other person - deletion needs > 50% support.
#   b) Competing rewrites -> similar rewrites are CLUSTERED together
#      (two people fixing the same typo differently end up in one
#      cluster), then clusters are ranked by number of supporters.
#   c) Tie-break 1: editor trust score (average trust of supporters).
#   d) Tie-break 2: the most conservative edit wins (highest similarity
#      to the original sentence) - minimal-change principle.
# ======================================================================

def similarity(a: str, b: str) -> float:
    """0..1 similarity between two sentences."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def cluster_rewrites(rewrites, trust, threshold=0.75):
    """
    Group near-identical rewrites into clusters.
    Two editors fixing the same typo slightly differently (similarity >= 0.75)
    land in the same cluster and their votes are POOLED. The cluster's
    representative text is the version written by the MOST TRUSTED supporter.
    Returns: list of clusters, each {"text": representative, "supporters": [...]}
    """
    clusters = []
    for rw in rewrites:
        placed = False
        for c in clusters:
            if similarity(rw["text"], c["text"]) >= threshold:
                c["supporters"].append(rw["editor"])
                # most trusted editor's wording represents the cluster
                if trust.get(rw["editor"], 1.0) > trust.get(c["author"], 1.0):
                    c["text"], c["author"] = rw["text"], rw["editor"]
                placed = True
                break
        if not placed:
            clusters.append({"text": rw["text"], "supporters": [rw["editor"]], "author": rw["editor"]})
    return clusters


def resolve_sentence(original, edits, trust):
    """
    Resolve all edits on ONE sentence. Returns (final_text_or_None, reason).
    final_text = None means the sentence was deleted by majority vote.
    """
    if not edits:
        return original, "untouched - kept as is"

    voters = {e["editor"] for e in edits}
    delete_votes = [e["editor"] for e in edits if e["action"] == "delete"]

    # (a) Deletion requires a strict majority of everyone who touched it
    if len(delete_votes) > len(voters) / 2:
        return None, f"DELETED - majority vote ({len(delete_votes)}/{len(voters)} editors voted delete)"
    elif delete_votes:
        note = f"deletion by {', '.join(delete_votes)} REJECTED ({len(delete_votes)}/{len(voters)} is not a majority); "
    else:
        note = ""

    # (b) Cluster the competing rewrites and count pooled votes
    rewrites = [e for e in edits if e["action"] == "rewrite"]
    keeps = [e["editor"] for e in edits if e["action"] == "keep"]
    if not rewrites:
        return original, note + "no surviving rewrite - original kept"

    clusters = cluster_rewrites(rewrites, trust)
    # "keep original" also competes as a candidate, supported by keep-voters
    clusters.append({"text": original, "supporters": keeps, "is_original": True})

    def score(c):
        votes = len(c["supporters"])
        avg_trust = (sum(trust.get(s, 1.0) for s in c["supporters"]) / votes) if votes else 0.0
        conservatism = similarity(c["text"], original)
        # lexicographic ranking: votes, then trust, then minimal change
        return (votes, avg_trust, conservatism)

    clusters.sort(key=score, reverse=True)
    winner, runner_up = clusters[0], (clusters[1] if len(clusters) > 1 else None)

    # Build a human-readable explanation of WHY this version won
    v_w = len(winner["supporters"])
    if runner_up and len(runner_up["supporters"]) == v_w:
        t_w = sum(trust.get(s, 1.0) for s in winner["supporters"]) / max(v_w, 1)
        t_r = sum(trust.get(s, 1.0) for s in runner_up["supporters"]) / max(v_w, 1)
        if t_w != t_r:
            why = f"tie on votes ({v_w} each) -> broken by trust score ({t_w:.2f} vs {t_r:.2f})"
        else:
            why = f"tie on votes and trust -> most conservative edit wins (closest to original)"
    else:
        why = f"won the vote ({v_w} supporter(s): {', '.join(winner['supporters']) or 'original text'})"

    if winner.get("is_original"):
        return original, note + "original kept - " + why
    return winner["text"], note + why


def merge_document(data):
    trust = data.get("trust_scores", {})
    print("=" * 72)
    print("DOCUMENT MERGER - resolving the five-cooks draft")
    print("=" * 72)
    final = []
    for i, sent in enumerate(data["document"]):
        edits = [e for e in data["edits"] if e["sentence"] == i]
        result, reason = resolve_sentence(sent, edits, trust)
        print(f"\nSentence {i}: \"{sent}\"")
        for e in edits:
            detail = f' -> "{e["text"]}"' if e["action"] == "rewrite" else ""
            print(f"   - {e['editor']}: {e['action'].upper()}{detail}")
        print(f"   DECISION: {reason}")
        if result is not None:
            print(f"   FINAL:    \"{result}\"")
            final.append(result)
    print("\n" + "-" * 72)
    print("FINAL MERGED DOCUMENT (one voice, every conflict resolved):\n")
    print(" ".join(final))
    return final


# ======================================================================
# PROBLEM 2: THE TEAM ROLE OPTIMIZER
# ----------------------------------------------------------------------
# Decision logic: this is the classic ASSIGNMENT PROBLEM.
# Build an 11x11 skill matrix (players x role slots) and let the
# Hungarian algorithm (scipy.linear_sum_assignment) pick the one-to-one
# assignment that maximizes total team skill. A tiny preference bonus
# breaks ties in favour of what a player wants - preferences matter,
# but only when the team doesn't pay for it.
# ======================================================================

FORMATION = ["GK", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "FWD", "FWD"]


def assign_team(data):
    players = data["players"]
    n = len(players)
    slots = data.get("formation", FORMATION)[:n]

    print("\n" + "=" * 72)
    print("TEAM ROLE OPTIMIZER - everyone wants to be the striker")
    print("=" * 72)

    # Skill matrix: rows = players, cols = formation slots
    M = np.zeros((n, len(slots)))
    for i, p in enumerate(players):
        for j, role in enumerate(slots):
            M[i, j] = p["skills"][role]
            if p.get("preference") == role:
                M[i, j] += 0.1  # tiny bonus: preference is only a tie-breaker

    # Hungarian algorithm: maximize total skill (minimize negative skill)
    rows, cols = linear_sum_assignment(-M)
    optimal_total = sum(players[i]["skills"][slots[j]] for i, j in zip(rows, cols))

    # The "too many cooks" baseline: everyone plays their PREFERRED role
    naive_total = sum(p["skills"][p["preference"]] for p in players)

    print(f"\n{'Player':<10}{'Wants':<8}{'Assigned':<10}{'Skill there':<12}{'Skill at wish'}")
    print("-" * 55)
    for i, j in sorted(zip(rows, cols), key=lambda x: slots[x[1]]):
        p = players[i]
        role = slots[j]
        mark = "  <-- got their wish" if p["preference"] == role else ""
        print(f"{p['name']:<10}{p['preference']:<8}{role:<10}{p['skills'][role]:<12}{p['skills'][p['preference']]}{mark}")

    print("-" * 55)
    print(f"Team skill if EVERYONE plays their preferred role : {naive_total}")
    print(f"Team skill with OPTIMAL assignment                : {optimal_total}")
    gain = optimal_total - naive_total
    print(f"=> The team gains {gain} skill points ({gain / naive_total:+.0%}) just by")
    print("   putting each cook at their own station instead of all at the stove.")
    return dict(zip([players[i]["name"] for i in rows], [slots[j] for j in cols]))


# ======================================================================
# DEMO DATA - realistic, editable, NOT single-user
# ======================================================================

DEMO_EDITS = {
    "trust_scores": {"Alice": 0.9, "Bob": 0.7, "Carol": 0.8, "Dan": 0.6, "Eve": 0.75},
    "document": [
        "Our compny was founded in 2015.",
        "We build software for small businesses.",
        "The team is passionate and hardworking.",
        "Customer satisfaction is our top priority.",
    ],
    "edits": [
        # Two people fix the same typo differently -> clustered, votes pooled
        {"editor": "Alice", "sentence": 0, "action": "rewrite", "text": "Our company was founded in 2015."},
        {"editor": "Bob",   "sentence": 0, "action": "rewrite", "text": "Our Company was founded in 2015."},
        {"editor": "Carol", "sentence": 0, "action": "rewrite", "text": "Founded in 2015, we are a young firm."},
        # One deletes a paragraph another just polished -> deletion outvoted
        {"editor": "Dan",   "sentence": 2, "action": "delete"},
        {"editor": "Eve",   "sentence": 2, "action": "rewrite", "text": "Our team is passionate and hardworking."},
        {"editor": "Alice", "sentence": 2, "action": "keep"},
        # Everyone rewrites in their own style -> vote, then trust tie-break
        {"editor": "Bob",   "sentence": 3, "action": "rewrite", "text": "We put customers first, always."},
        {"editor": "Carol", "sentence": 3, "action": "rewrite", "text": "Customer happiness drives everything we do."},
        {"editor": "Dan",   "sentence": 3, "action": "rewrite", "text": "We put customers first always."},
    ],
}

DEMO_PLAYERS = {
    "players": [
        {"name": "Marco",  "preference": "FWD", "skills": {"GK": 40, "DEF": 55, "MID": 70, "FWD": 92}},
        {"name": "Luis",   "preference": "FWD", "skills": {"GK": 35, "DEF": 60, "MID": 75, "FWD": 88}},
        {"name": "Kofi",   "preference": "FWD", "skills": {"GK": 30, "DEF": 85, "MID": 65, "FWD": 60}},
        {"name": "Sergio", "preference": "FWD", "skills": {"GK": 25, "DEF": 90, "MID": 60, "FWD": 55}},
        {"name": "Yuto",   "preference": "FWD", "skills": {"GK": 20, "DEF": 82, "MID": 70, "FWD": 58}},
        {"name": "Pavel",  "preference": "FWD", "skills": {"GK": 30, "DEF": 88, "MID": 55, "FWD": 50}},
        {"name": "Amir",   "preference": "FWD", "skills": {"GK": 25, "DEF": 65, "MID": 90, "FWD": 70}},
        {"name": "Jonas",  "preference": "FWD", "skills": {"GK": 20, "DEF": 60, "MID": 87, "FWD": 68}},
        {"name": "Diego",  "preference": "FWD", "skills": {"GK": 15, "DEF": 55, "MID": 85, "FWD": 72}},
        {"name": "Theo",   "preference": "FWD", "skills": {"GK": 30, "DEF": 62, "MID": 83, "FWD": 66}},
        {"name": "Ivan",   "preference": "FWD", "skills": {"GK": 93, "DEF": 50, "MID": 40, "FWD": 35}},
    ]
}


def main():
    args = sys.argv[1:]
    if not args or args[0] == "demo":
        merge_document(DEMO_EDITS)
        assign_team(DEMO_PLAYERS)
    elif args[0] == "merge":
        with open(args[1]) as f:
            merge_document(json.load(f))
    elif args[0] == "assign":
        with open(args[1]) as f:
            assign_team(json.load(f))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
