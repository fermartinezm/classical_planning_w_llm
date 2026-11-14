from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether a fact string matches a pattern with wildcards.

    Args:
        fact: A PDDL fact string, e.g., '(tightened nut1 locatable13)'
        args: Pattern parts to match against, e.g. 'tightened', '*', 'locatable13'

    Returns:
        True if matches, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'spanner' (secret_domain) domain.

    Summary:
    ----------
    This heuristic estimates the minimal number of required actions to achieve
    the goal in states where nuts must be tightened at specific locatable objects
    and locations. The heuristic is designed to guide a greedy best-first search
    by counting how many nuts are not yet at their goal tightened locations
    plus an estimate of partial effort to move and tighten them correctly.

    Assumptions:
    ---------------
    - Each nut starts "tightened" on some locatable object, and must be moved/tightened
      to a goal locatable + location combination given by "tightened-2" facts in the goal.
    - Nuts can be loose or tightened at locatables.
    - The "link" facts indicate which locatable objects can be used to move/transport/tighten.
    - The final goal requires certain locatables to be tightened at specific locations.
    - Actions include tightening nuts and "walking" or "picking up" spanners/tools associated with locatables.
    - The heuristic does not need to be admissible but should reflect how many nuts remain
      to be moved/tightened to the goal to guide search efficiency.

    Heuristic Initialization:
    --------------------------
    - Extract all goal facts of the form (tightened-2 <locatable> <location>).
      These facts indicate where each locatable must be tightened at the end.
    - Extract static facts that indicate:
        * which locatables are linked (link predicate),
        * loose relations between locatables (loose predicate),
        * usable relations between locatables (usable predicate),
        * carrying relations (carrying predicate).
      These are used for basic connectivity and reachability approximations.
    - No dynamic state info is stored here; only static connectivity info.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify all goal (tightened-2 locatable location) pairs:
       Each such locatable must be tightened at the specified location in the goal.

    2. For each such locatable goal:
       a) Check if the current state already has (tightened-2 locatable location).
          If yes, it contributes zero cost.
       b) Otherwise, estimate the cost to "move" that locatable from its current tightened
          locatable to the goal location via the graph of link, loose, carrying, and usable relations.
          Because exact shortest paths are expensive, edges can be built from static facts:
            - Links between locatables represent movement edges.
            - Usable edges represent tightening/spanner movement possibilities.
            - Loose relations allow loosening/tightening actions.

    3. Estimate the minimal number of "move" steps (hops) to get the locatable tightened at the goal location.
       We will build a graph where nodes are locatables and/or locations, edges represent adjacency.
       To handle mixed locatable-location predicates:
          - Use locatables as nodes.
          - Use locations implicitly by checking tightened-2 facts.

    4. For each goal locatable:
       Find its current tightened position locatable (the locatable where it is tightened now).
       Then estimate the shortest "path" (by number of steps) to move/tighten it at the goal location.

    5. Each movement or tightening corresponds roughly to an action.
       So the heuristic sums shortest path hops plus one for the final tightening action.

    6. If a locatable currently is not tightened where required (i.e., no tightened-2 goal fact matched),
       the heuristic assumes at least one move/tighten is needed for that locatable.

    7. The heuristic sums these estimates over all goal locatables.

    Notes:
    - This heuristic ignores spanner carrying subtleties and actual action sequences for efficiency.
    - Being non-admissible, it may overestimate or underestimate actual cost but guides the search effectively.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal (tightened-2 locatable location) facts:
        self.goal_tightened_2 = {}  # locatable -> location
        for gt in self.goals:
            parts = get_parts(gt)
            if parts[0] == "tightened-2" and len(parts) == 3:
                locatable, location = parts[1], parts[2]
                self.goal_tightened_2[locatable] = location

        # Extract static facts relevant for connectivity and movement:
        # We treat locatables as nodes in a graph.
        self.links = set()
        self.loose = set()
        self.usable = set()
        self.carrying = set()

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "link" and len(parts) == 2:
                self.links.add(parts[1])
            elif pred == "loose" and len(parts) == 3:
                self.loose.add((parts[1], parts[2]))
            elif pred == "usable" and len(parts) == 3:
                self.usable.add((parts[1], parts[2]))
            elif pred == "carrying" and len(parts) == 3:
                self.carrying.add((parts[1], parts[2]))

        # Build adjacency dict for locatables based on static relations:
        # We'll create an undirected connectivity graph between locatables,
        # which summarizes movement possibilities:
        self.adjacency = {}  # locatable -> set(locatable)

        def add_edge(a, b):
            if a not in self.adjacency:
                self.adjacency[a] = set()
            if b not in self.adjacency:
                self.adjacency[b] = set()
            self.adjacency[a].add(b)
            self.adjacency[b].add(a)

        # Add edges from link facts (locatables linked directly)
        for loc in self.links:
            # We don't have pairs here, links are unary predicates.
            # So links alone can't generate edges, but
            # loose, carrying, usable are between locatables.

            # We'll handle edges between locatables with loose, carrying, usable:
            pass

        # Add edges from loose relations
        for a, b in self.loose:
            add_edge(a, b)

        # Add edges from carrying relations
        for a, b in self.carrying:
            add_edge(a, b)

        # Add edges from usable relations
        for a, b in self.usable:
            add_edge(a, b)

    def __call__(self, node):
        state = node.state

        # Quick check: if goal reached, heuristic=0
        if self.goals <= state:
            return 0

        # Helper: find current tightens (tightened locatable for each nut)
        # Facts of type: (tightened nut locatable)
        tightened = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "tightened" and len(parts) == 3:
                nut, locatable = parts[1], parts[2]
                tightened[nut] = locatable

        # Facts of type (tightened-2 locatable location) indicate locatables tightened at location
        tightened_2 = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "tightened-2" and len(parts) == 3:
                locatable, location = parts[1], parts[2]
                tightened_2[locatable] = location

        # We want to estimate cost to move each goal locatable to the goal location.
        # For each goal locatable, check if it's already tightened-2 at the goal location.
        # If not, estimate minimal steps:

        # Function: BFS shortest path between two locatables on adjacency graph
        def shortest_path_length(start, goal):
            if start == goal:
                return 0
            if start not in self.adjacency or goal not in self.adjacency:
                return 1000  # Large penalty for disconnected nodes
            from collections import deque

            visited = set([start])
            queue = deque([(start, 0)])
            while queue:
                current, dist = queue.popleft()
                for nbr in self.adjacency.get(current, []):
                    if nbr == goal:
                        return dist + 1
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, dist + 1))
            return 1000  # No path found

        heuristic = 0

        # We don't have explicit relationship nut<->locatable in the goals,
        # but the goal is tightened-2 locatable location for locatable objects.
        # However, nuts are tightened on locatables. So cross-reference nuts and locatables.

        # Since goal only fixes (tightened-2 locatable location),
        # for every locatable in goal_tightened_2, check current tightened-2 status:
        for locatable_goal, loc_goal_loc in self.goal_tightened_2.items():
            # If already at goal tightened-2, zero cost:
            curr_loc = tightened_2.get(locatable_goal)
            if curr_loc == loc_goal_loc:
                continue  # no cost needed

            # Otherwise, estimate cost to get locatable_goal tightened at loc_goal_loc

            # Find where the locatable_goal currently is tightened-2 (if any)
            # or assume it is at some generic location if none found.
            # We'll consider states achieving tightens on locatables by moving between them.

            # Get all locatables that currently contain tightened nut corresponding to locatable_goal
            # This is heuristic guess: nuts are tightened on locatables,
            # so find the nut tightened on the locatable_goal and use its position.

            # But locatable_goal is a locatable. We must find associated nut tightened on it.

            # Search nuts that currently tightened to locatable_goal
            nuts_here = [nut for nut, loc in tightened.items() if loc == locatable_goal]

            if not nuts_here:
                # No nut tightened on this locatable in current state,
                # heuristic assumes at least 2 moves to fix (arbitrary penalty)
                heuristic += 5
                continue

            # For each such nut, find where it is currently tightened (locatable)
            # We will pick the first (since typically one nut per locatable)
            nut = nuts_here[0]
            curr_loc_on = tightened.get(nut, None)  # locatable nut currently tightened on
            if curr_loc_on is None:
                # No info, guess large cost
                heuristic += 5
                continue

            # Currently, locatable_goal is to be tightened at loc_goal_loc,
            # but in the current state tightened_2 may be missing or different.

            # We want to estimate the minimal action cost to move/tighten nut from curr_loc_on to locatable_goal,
            # then move locatable_goal to loc_goal_loc by tightening.

            # Step 1: Distance from curr_loc_on to locatable_goal in adjacency (representing how to move nut)
            dist_move_nut = shortest_path_length(curr_loc_on, locatable_goal)

            # Step 2: If current locatable_goal is not tightened-2 at loc_goal_loc (goal), must move it:
            # Estimate distance from current loosened location (if known) to goal location.

            # Find current location of locatable_goal in tightened-2 facts or guess.
            curr_goal_loc = tightened_2.get(locatable_goal, None)
            if curr_goal_loc is None:
                # We do not know current location for tightened-2 locatable,
                # but locations are from (at-2 location) facts in static.
                # As a fallback, cost 1 to reach goal location:
                dist_to_goal_loc = 1
            else:
                # If not at goal location, distance 1 (need tightening action)
                dist_to_goal_loc = 0 if curr_goal_loc == loc_goal_loc else 1

            # Total cost is sum of:
            # - moving nut from curr_loc_on to locatable_goal (approximate as dist_move_nut)
            # - tightening locatable_goal at loc_goal_loc (always at least 1 action if not already)
            # - plus 1 for any final tightened-2 adjustment if location differs

            loc_cost = dist_move_nut + dist_to_goal_loc + 1  # +1 for tightening action

            heuristic += loc_cost

        # Ensure heuristic is finite and >= 0
        return max(heuristic, 1)
