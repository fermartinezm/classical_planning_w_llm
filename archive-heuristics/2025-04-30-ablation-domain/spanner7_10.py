from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It considers the cost of tightening
    each nut, the movement cost for Bob to visit all necessary nut locations,
    and the cost for Bob to acquire a spanner if he isn't already carrying one.

    # Assumptions
    - Bob can only carry one spanner at a time.
    - All spanners marked as 'usable' remain usable throughout the plan.
    - Links between locations are bidirectional.
    - There is always a usable spanner available on the ground if Bob isn't carrying one (solvable problem assumption).

    # Heuristic Initialization
    - Build the graph of locations and links from static facts to enable distance calculations.
    - Identify all nuts that need to be tightened from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Loose Goal Nuts: Find all nuts that are currently loose in the state
       and are required to be tightened in the goal. Iterate through the state facts
       to find `(loose ?nut)` predicates. Check if `?nut` is in the set of nuts
       identified from the goal conditions. If no loose goal nuts are found, the
       heuristic is 0 (goal state or part of goal state achieved).
    2. Base Cost: Initialize the heuristic value with the number of loose goal nuts.
       Each loose goal nut requires at least one 'tighten' action.
    3. Find Bob's Location: Search the state facts for `(at bob ?location)`.
       If Bob's location cannot be determined, the state is likely unsolvable,
       return infinity.
    4. Calculate Distances: Perform a Breadth-First Search (BFS) starting from Bob's
       current location (`LocBob`) on the location graph built during initialization.
       This calculates the shortest path distance (minimum number of 'move' actions)
       from `LocBob` to all other reachable locations. Store these distances.
    5. Movement Cost: Identify the current location for each loose goal nut by
       searching the state facts for `(at ?nut ?location)`. Collect the set of
       unique locations where loose goal nuts are found. For each unique nut
       location (`LocN`) that is different from Bob's current location (`LocBob`),
       add the shortest distance from `LocBob` to `LocN` (obtained from the BFS)
       to the heuristic. If any required nut location is unreachable, the state
       is unsolvable, return infinity. This sum estimates the travel needed to
       visit all locations where work is required.
    6. Spanner Acquisition Cost: Check if Bob is currently carrying any object
       by searching for `(carrying bob ?obj)` in the state facts.
       - If Bob is not carrying a spanner: He needs to acquire one. First, identify
         all usable spanners by searching for `(usable ?spanner)` facts in the state.
         Then, find the locations of these usable spanners by searching for
         `(at ?spanner ?location)` facts. If no usable spanner is found on the
         ground, the state is likely unsolvable, return infinity. Otherwise, find
         the minimum shortest distance from Bob's current location to any location
         containing a usable spanner on the ground. Add this minimum distance plus
         1 (for the 'pickup' action) to the heuristic. If all usable spanner
         locations are unreachable, return infinity.
       - If Bob is carrying a spanner: The cost to acquire a spanner is 0, as he
         already has one.
    7. Total Heuristic: The final heuristic value is the sum accumulated from
       steps 2, 5, and 6.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the location graph.
        """
        self.goals = task.goals

        # Identify nuts that need to be tightened from the goal conditions
        self.nuts_to_tighten = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                self.nuts_to_tighten.add(args[0])

        # Build the location graph from static link facts
        self.graph = {}
        locations = set()

        # Collect all locations mentioned in static facts and initial state
        # Look for arguments in 'at' and 'link' predicates
        facts_to_scan = task.static | task.initial_state
        for fact in facts_to_scan:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 # (at obj loc) - loc is a location
                 locations.add(parts[2])
             elif parts[0] == "link" and len(parts) == 3:
                 # (link loc1 loc2) - loc1 and loc2 are locations
                 locations.add(parts[1])
                 locations.add(parts[2])

        self.graph = {loc: [] for loc in locations}

        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Add links only if both locations were identified
                if loc1 in self.graph and loc2 in self.graph:
                    self.graph[loc1].append(loc2)
                    self.graph[loc2].append(loc1) # Links are bidirectional

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find shortest distances to all reachable nodes.
        Returns a dictionary mapping locations to their distance from start_node.
        Returns float('inf') for unreachable locations.
        """
        # Initialize distances for all known locations
        distances = {node: float('inf') for node in self.graph}

        # If the start node is not in our graph of known locations,
        # it's an isolated location not linked to anything we know.
        # Distance to itself is 0, others remain inf.
        if start_node not in distances:
             distances[start_node] = 0
             return distances # Cannot reach other nodes from here

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Ensure u is still a valid node in the graph (should be if it came from queue)
            if u in self.graph:
                for v in self.graph[u]:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Loose Goal Nuts
        loose_goal_nuts = set()
        for nut in self.nuts_to_tighten:
             if f"(loose {nut})" in state:
                  loose_goal_nuts.add(nut)

        # 2. Base Cost
        if not loose_goal_nuts:
            return 0 # Goal reached

        total_cost = len(loose_goal_nuts) # Cost for tighten actions

        # 3. Find Bob's Location
        bob_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_location = get_parts(fact)[2]
                break
        if bob_location is None:
             # Bob's location is unknown, problem likely unsolvable from here
             return float('inf')

        # 4. Calculate Distances from Bob's Location
        distances_from_bob = self._bfs(bob_location)

        # 5. Movement Cost
        nut_locations = {}
        for nut in loose_goal_nuts:
             found_loc = None
             for fact in state:
                  if match(fact, "at", nut, "*"):
                       found_loc = get_parts(fact)[2]
                       break # Assuming nut is only at one location
             if found_loc:
                 nut_locations[nut] = found_loc
             else:
                 # A loose goal nut is not located anywhere? Should not happen in valid states.
                 # Treat as unsolvable from here.
                 return float('inf')


        unique_nut_locations = set(nut_locations.values())

        # Sum distances to unique nut locations Bob is not currently at
        movement_cost = 0
        for loc in unique_nut_locations:
             if loc != bob_location:
                  dist = distances_from_bob.get(loc, float('inf'))
                  if dist == float('inf'):
                       # A nut location is unreachable from Bob's current location
                       return float('inf') # Unsolvable state
                  movement_cost += dist

        total_cost += movement_cost

        # 6. Spanner Acquisition Cost
        carrying_spanner = any(match(fact, "carrying", "bob", "*") for fact in state)

        if not carrying_spanner:
            # Find all usable spanners
            usable_spanners = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

            # Find locations of usable spanners on the ground
            usable_spanner_locations = set()
            for spanner in usable_spanners:
                 for fact in state:
                      if match(fact, "at", spanner, "*"):
                           usable_spanner_locations.add(get_parts(fact)[2])
                           break # Assuming spanner is only at one location

            if not usable_spanner_locations:
                # No usable spanners available on the ground and not carrying one
                # This state is likely unsolvable unless a spanner can be created/repaired (not in domain)
                return float('inf') # Unsolvable state

            min_dist_to_spanner = float('inf')
            for loc in usable_spanner_locations:
                 dist = distances_from_bob.get(loc, float('inf'))
                 if dist != float('inf'):
                      min_dist_to_spanner = min(min_dist_to_spanner, dist)

            if min_dist_to_spanner == float('inf'):
                 # All usable spanner locations are unreachable
                 return float('inf') # Unsolvable state

            spanner_acquisition_cost = min_dist_to_spanner + 1 # Move to spanner + pickup
            total_cost += spanner_acquisition_cost

        # 7. Total Heuristic
        return total_cost
