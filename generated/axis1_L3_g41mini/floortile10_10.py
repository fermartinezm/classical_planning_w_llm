from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components from a PDDL fact string, removing surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a pattern with wildcards.

    Args:
        fact (str): The fact string, e.g. "(tightened nut1 locatable13)".
        *args (str): Pattern parts to match, e.g. "tightened", "*", "locatable13".

    Returns:
        bool: True if matches, False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (spanner) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve all
    goal conditions of the form (tightened-2 <locatable> <location>) from
    the current state. It leverages the domain structure around nuts and
    locatables that must be "tightened" at specific locations.

    Assumptions
    -----------
    - The goal always requires certain locatables to be tightened at certain locations,
      expressed as (tightened-2 locatable location).
    - Nuts are associated with locatable objects and can be loose or tightened.
    - Actions typically move tightening status of nuts linked to locatables and locations.
    - The static facts define the adjacency and movability relations (loose, at, usable, carrying, link).
    - Moving a tightened nut/linkable chain along locatable links changes which locatable/nut pair is tightened.
    - The heuristic aims at approximating the minimal number of required moves to move tightened objects
      to their goal location, plus the effort to tighten loose nuts at their locations.

    Heuristic Initialization
    ------------------------
    - Extract goal (tightened-2 locatable location) facts, mapping locatable -> goal location.
    - Extract static world info:
      * "loose" edges between locatables defining adjacency graph.
      * Facts indicating which nuts correspond to which locatables and are loose/tightened.
      * "usable" relations indicating possible chains of locatables to move tightening.
      * Carrying relations for potential shortcuts.
      * Locations in the domain.
    - Build a graph representing adjacencies of locatables.
    - Record initial nut-to-locatable tightening relations.
    - Record loose nuts to locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if state is a goal state; if yes, heuristic=0.
    2. For each goal (tightened-2 locatable L at location Loc_goal):
       - Identify which nut is currently tightened at that locatable (if any).
       - If the tightened chain is already at the goal location (tightened-2 locatable Loc_goal),
         no cost for that particular goal.
       - Otherwise, estimate distance cost to move the tightened nut/link to goal location.
         Distance is approximated by shortest path length in the locatable adjacency graph.
    3. For each loose nut at some location not at its goal location:
       - Include cost for tightening it at the correct location (each loosened nut must be tightened).
    4. Sum these distances and actions estimations for all locatables appearing in goals.
    5. Also consider that moving or tightening likely costs about 1 action each,
       and realistically, a sequence of actions per step is needed (e.g., pick, walk, tighten).
    6. Return the sum as heuristic estimate; it is not guaranteed admissible but aims to be informative.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts e.g. (tightened-2 locatable location)
        static = task.static

        # Extract goal locatable -> goal location mapping
        self.goal_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened-2" and len(parts) == 3:
                locatable = parts[1]
                location = parts[2]
                self.goal_map[locatable] = location

        # Build adjacency graph of locatables from loose/at relations (static)
        self.graph = {}
        self.loose_edges = set()
        for fact in static:
            if match(fact, "loose", "*", "*"):  # (loose loc1 loc2)
                _, loc1, loc2 = get_parts(fact)
                self.loose_edges.add((loc1, loc2))
                self.loose_edges.add((loc2, loc1))
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)

            if match(fact, "at", "*", "*"):  # (at loc1 loc2), to augment graph connections
                _, loc1, loc2 = get_parts(fact)
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)

        # Extract all locatables (from graph keys) and locations
        # Locations are objects that appear in at-2 facts in static.
        self.locations = set()
        for fact in static:
            if fact.startswith("(at-2 "):
                parts = get_parts(fact)
                if len(parts) == 2:
                    self.locations.add(parts[1])
                elif len(parts) == 3:
                    self.locations.add(parts[2])

        # Extract loose nuts with their locations (from loose-2 nuts loc)
        self.loose_nuts = {}
        for fact in static:
            if match(fact, "loose-2", "*", "*"):
                _, nut, loc = get_parts(fact)
                self.loose_nuts[nut] = loc

        # For efficient nut-to-locatable relation, store types and track nuts names, locatables
        self.nuts = set()
        self.locatables = set(self.graph.keys())
        for fact in static:
            if fact.startswith("(tightened "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    nut = parts[1]
                    if nut not in self.nuts:
                        self.nuts.add(nut)

        # Precompute shortest paths between any two locatables using BFS since graphs are not large
        # We'll do this in the heuristic call since state differs, but limited for efficiency
        # Instead, heuristic will compute BFS on demand for goal locatables.

    def shortest_path_length(self, start, goal):
        """Compute an approximate shortest path length between locatables start and goal."""
        if start == goal:
            return 0
        if start not in self.graph or goal not in self.graph:
            return 10**6  # Large number if path unavailable
        from collections import deque
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current == goal:
                return dist
            visited.add(current)
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
                    visited.add(neighbor)
        return 10**6  # No path found

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract tightened locatable-to-nut mappings in current state: tightened(nut locatable)
        tightened_loc_to_nut = {}
        for fact in state:
            if match(fact, "tightened", "*", "*"):
                _, nut, locatable = get_parts(fact)
                tightened_loc_to_nut[locatable] = nut

        # Extract loosened nuts and their current loose location (loose-2 nut location)
        loose_nut_locations = {}
        for fact in state:
            if match(fact, "loose-2", "*", "*"):
                _, nut, loc = get_parts(fact)
                loose_nut_locations[nut] = loc

        # Extract locations where locatables currently are tightened-2 locatable location
        tightened_2_facts = {}
        for fact in state:
            if match(fact, "tightened-2", "*", "*"):
                _, locatable, location = get_parts(fact)
                tightened_2_facts[locatable] = location

        # Estimate cost to achieve all goal tightened-2 locatable location facts.
        heuristic_cost = 0

        for locatable_goal, goal_location in self.goal_map.items():
            current_location = tightened_2_facts.get(locatable_goal, None)

            # Case 1: the locatable is already tightened at the goal location
            if current_location == goal_location:
                continue  # no cost needed for this locatable

            # Case 2: otherwise, estimate movement cost for tightening chain
            # Identify nut currently tightened at locatable_goal, else it must be tightened at some other locatable
            nut = tightened_loc_to_nut.get(locatable_goal, None)
            if nut is None:
                # No nut tightened at locatable_goal => must tighten loose nut and move it here.
                # Check if loose nut exists for this nut name
                # We consider heuristically that tightening loose nut costs at least 1 action + move

                # Find loose nuts that belong or can be tightened here (must match nut)
                # Otherwise, count cost to move loose nut (from loose location to goal location)
                # Since we do not know nut for locatable_goal, count minimal cost: at least 2 actions (tighten + move)

                heuristic_cost += 3  # heuristic fixed cost: tighten + move (approx)
                continue

            # Nut is tightened at locatable_goal, but possibly at wrong place.
            # Find current location of nut (tightened-2 nut location)
            nut_loose_loc = loose_nut_locations.get(nut, None)  # loose nut location (if any)

            # Distance to move from current tightened-2 locatable position to goal location
            if current_location is None:
                # Not tightened anywhere now, must move from loose location or somewhere else
                if nut_loose_loc is not None:
                    dist = self.shortest_path_length(nut_loose_loc, goal_location)
                    heuristic_cost += dist + 1  # distance + tighten action
                else:
                    # Nut position unknown, assign penalty
                    heuristic_cost += 5
            else:
                # Move tightened locatable chain from current_location to goal_location
                dist = self.shortest_path_length(current_location, goal_location)
                heuristic_cost += dist

        # Also add cost for tightening loose nuts at their locations (if their tightening is a goal)
        for nut, loc in loose_nut_locations.items():
            # Check if the nut tightening is already satisfied in goal (at any locatable at correct location)
            # If nut must be tightened to achieve some tightened-2?
            # We estimate at least one action per loose nut to tighten it
            # This accounts for the "tighten_nut" actions (parameterized by nuts and locations)
            heuristic_cost += 1

        # Bound heuristic to finite number
        heuristic_cost = min(heuristic_cost, 10**6)

        return heuristic_cost
