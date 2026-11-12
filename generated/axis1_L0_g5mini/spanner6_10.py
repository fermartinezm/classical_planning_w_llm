from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the "spanner" domain estimating the number of actions required
    to tighten all remaining loose nuts.

    Assumptions
    -----------
    - Each usable spanner can be used exactly once (tighten_nut makes it unusable).
    - A spanner being carried by a man is only usable by that man (there is no drop action).
    - Men can carry multiple spanners simultaneously.
    - Movement cost (walk) between adjacent locations is 1 per link.
    - pickup_spanner costs 1 action, tighten_nut costs 1 action.
    - The heuristic need not be admissible; it should be informative and cheap to compute.
    - If distances between locations are unavailable (no connecting path), those assignments
      are treated as infeasible (very large cost).

    Heuristic Initialization
    ------------------------
    - Extracts the static link graph from task.static by reading (link L1 L2) facts
      and building an undirected adjacency structure for locations.
    - Stores the goal conditions (task.goals) to return zero when the goal is already reached.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goals are satisfied (all required tightened facts present), return 0.
    2. Parse the state to extract:
       - Locations of objects from (at OBJ LOC) facts.
       - Which spanners are usable from (usable SPANNER).
       - Which spanners are currently carried and by whom from (carrying MAN SPANNER).
       - Which nuts are loose from (loose NUT).
    3. Build a representation of usable spanners:
       - For each usable spanner, determine whether it is carried (and by whom) or located at
         some location. If carried, its effective position is the carrier's location; it is
         only available to that carrier.
       - If not carried, it is available to any man; its position is the recorded (at SPANNER LOC).
    4. Identify the men and their current positions using (carrying ...) and (at MAN LOC) facts.
    5. For each loose nut, greedily assign the currently best (man, usable-spanner) pair that
       can tighten it with minimal estimated action-cost. The cost model for using a spanner s
       by man m to tighten nut at location Ln is:
         - If s is currently carried by m:
             cost = distance(m_loc, Ln) + tighten_cost (1)
         - If s is not carried:
             cost = distance(m_loc, s_loc) + pickup_cost (1) + distance(s_loc, Ln) + tighten_cost (1)
       Distances are shortest-path lengths on the static link graph (BFS).
    6. After assigning a spanner to a nut, remove that spanner from the pool (it becomes unusable).
       Update the chosen man's position to be the nut's location (he ends up there after tightening).
    7. Sum the costs for all assignments. If some nut cannot be feasibly assigned (no path or no
       available spanner), add a large penalty to indicate high cost.
    8. Return the computed integer cost.

    Notes
    -----
    - This heuristic performs a greedy matching between nuts and usable spanners. It is optimistic
      in that it ignores some possible synergies (e.g., collecting multiple spanners in one trip),
      but it captures per-nut travel + pickup + tighten costs and thus guides greedy best-first search
      effectively in many instances.
    """

    def __init__(self, task):
        # Store goal conditions so we can return 0 on goal states.
        self.goals = task.goals

        # Build undirected adjacency from static (link ...) facts.
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # Keep a set of statically known locations (nodes in adjacency).
        self.static_locations = set(self.adj.keys())

    def shortest_distance(self, start, goal):
        """
        Compute shortest distance (number of walk actions) between two locations using BFS.
        If start == goal returns 0. If either node is missing in the adjacency or unreachable,
        returns float('inf').
        """
        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            # If a node does not appear in the static link graph, treat as unreachable.
            return float("inf")
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            loc, d = dq.popleft()
            for nbr in self.adj.get(loc, ()):
                if nbr == goal:
                    return d + 1
                if nbr not in visited:
                    visited.add(nbr)
                    dq.append((nbr, d + 1))
        return float("inf")

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse relevant facts from the state.
        at = {}  # subject -> location, for (at subj loc)
        carrying = {}  # man -> set(spanners)
        carried_spanner_owner = {}  # spanner -> man
        usable_spanners = set()  # set of spanner names that are usable
        loose_nuts = set()  # set of loose nut names

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]
            if pred == "at" and len(args) >= 2:
                subj, loc = args[0], args[1]
                at[subj] = loc
            elif pred == "carrying" and len(args) >= 2:
                man, sp = args[0], args[1]
                carrying.setdefault(man, set()).add(sp)
                carried_spanner_owner[sp] = man
            elif pred == "usable" and len(args) >= 1:
                usable_spanners.add(args[0])
            elif pred == "loose" and len(args) >= 1:
                loose_nuts.add(args[0])

        # Collect loose nuts with their locations.
        nuts = []
        for n in loose_nuts:
            nuts.append((n, at.get(n)))  # location might be None if missing

        if not nuts:
            return 0

        # Build usable spanner info: for each usable spanner determine (holder, location).
        # If carried -> holder is that man and location is the man's location.
        # If not carried -> holder is None and location is the recorded spanner location (from at).
        spanner_infos = {}
        for s in usable_spanners:
            holder = carried_spanner_owner.get(s, None)
            if holder is not None:
                sp_loc = at.get(holder)  # spanner effectively at carrier's location
            else:
                sp_loc = at.get(s)
            spanner_infos[s] = (holder, sp_loc)

        # Identify men and their positions. Start with carrying-first-arg.
        men = set(carrying.keys())
        # Also add any subject of an (at SUBJ LOC) fact that looks like a man:
        # a simple heuristic: if the subject is not a known spanner and not a nut and not a static location,
        # treat it as a man (this captures typical problems like 'bob').
        for subj, loc in at.items():
            if subj not in spanner_infos and subj not in loose_nuts and subj not in self.static_locations:
                men.add(subj)

        men_pos = {m: at.get(m) for m in men}

        # If no men or no usable spanners, return a large heuristic (unsolvable or expensive).
        if not men_pos or not spanner_infos:
            # Finite but large: number of nuts times a large factor.
            return 100000 * len(nuts)

        # Greedy assignment of spanners to nuts:
        # For each nut, choose the remaining spanner that yields minimal estimated actions,
        # remove that spanner from availability, and update the chosen man's position to the nut's location.
        remaining_spanners = dict(spanner_infos)
        total_cost = 0
        BIG = 100000  # penalty for infeasible assignments

        # Prefer nuts with known locations first (sort to place ones without loc last).
        nuts_sorted = sorted(nuts, key=lambda x: 0 if x[1] is not None else 1)

        for nut, nut_loc in nuts_sorted:
            if nut_loc is None:
                # Unknown nut location: extremely costly / infeasible
                total_cost += BIG
                continue

            best_cost = float("inf")
            best_sp = None
            best_man = None

            # Try each available spanner.
            for sp, (holder, sp_loc) in remaining_spanners.items():
                if holder is None:
                    # spanner is on the ground and available to any man
                    if sp_loc is None:
                        # spanner location unknown -> skip
                        continue
                    for man, man_loc in men_pos.items():
                        if man_loc is None:
                            continue
                        d1 = self.shortest_distance(man_loc, sp_loc)
                        d2 = self.shortest_distance(sp_loc, nut_loc)
                        if d1 == float("inf") or d2 == float("inf"):
                            continue
                        # cost = walk to spanner + pickup + walk to nut + tighten
                        cost = d1 + 1 + d2 + 1
                        if cost < best_cost:
                            best_cost = cost
                            best_sp = sp
                            best_man = man
                else:
                    # spanner is carried by 'holder' and only that holder can use it
                    man = holder
                    man_loc = men_pos.get(man)
                    if man_loc is None:
                        continue
                    d = self.shortest_distance(man_loc, nut_loc)
                    if d == float("inf"):
                        continue
                    # cost = walk to nut (if necessary) + tighten (no pickup)
                    cost = d + 1
                    if cost < best_cost:
                        best_cost = cost
                        best_sp = sp
                        best_man = man

            if best_sp is None:
                # No feasible spanner for this nut -> big penalty
                total_cost += BIG
            else:
                # Use the chosen spanner for this nut.
                total_cost += best_cost
                # Remove spanner from pool (it becomes unusable after use).
                remaining_spanners.pop(best_sp, None)
                # Update chosen man's position to the nut location (he ends up there).
                if best_man is not None:
                    men_pos[best_man] = nut_loc

        # Return finite integer heuristic estimate.
        # Cap to integer to avoid returning floats like inf.
        if total_cost == float("inf"):
            return BIG * len(nuts)
        return int(total_cost)