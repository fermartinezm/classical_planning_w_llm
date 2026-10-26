from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Assume Heuristic base class is available and imported as Heuristic
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It considers the number of nuts
    to tighten, the cost of picking up spanners, and the travel cost
    to reach the nuts and spanners.

    # Assumptions
    - All nuts that need tightening are initially loose and at fixed locations.
    - Spanners are initially at fixed locations and usable, and become unusable after one use.
    - The man can only carry one spanner at a time (implied by the domain).
    - The location graph is connected.
    - Action costs are uniform (implicitly 1).

    # Heuristic Initialization
    - Identify all locations from the initial state and static facts.
    - Parse the `link` facts to build an adjacency list representation of the location graph.
    - Compute all-pairs shortest paths between locations using Breadth-First Search (BFS).
    - Identify the nuts that are part of the goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are in the goal state and are currently loose.
    3. Identify all usable spanners and their current locations (either on the ground or being carried by the man).
    4. Determine if the man is currently carrying a usable spanner.
    5. Count the number of loose goal nuts (`N_nuts_to_tighten`).
    6. Count the number of usable spanners available (`N_usable_spanners`).
    7. If `N_nuts_to_tighten` is 0, the heuristic is 0 (goal state or all goal nuts tightened).
    8. If `N_nuts_to_tighten > N_usable_spanners`, the state is likely unsolvable (requires more usable spanners than available), return a large value (infinity).
    9. Otherwise, the heuristic is calculated as the sum of:
        - The number of `tighten_nut` actions required (`N_nuts_to_tighten`).
        - The number of `pickup_spanner` actions required (`N_nuts_to_tighten`, assuming a new spanner must be picked up for each nut).
        - An estimate of the travel cost. A simple estimate is the sum of shortest path distances from the man's current location to the location of each loose goal nut, plus the sum of shortest path distances from the man's current location to the location of each usable spanner. This overestimates travel but captures the need to reach these objects.

    10. Final Heuristic Formula:
        Heuristic = `N_nuts_to_tighten` (tighten actions)
                   + `N_nuts_to_tighten` (pickup actions)
                   + Sum_{n in LooseGoalNuts} Distance(man_loc, ObjLocs[n]) # Travel to nuts
                   + Sum_{s in UsableSpanners} Distance(man_loc, ObjLocs[s]) # Travel to spanners

        This formula sums the minimum number of actions (tighten, pickup) and adds the sum of straight-line distances (shortest path) from the man's current location to all relevant objects (loose goal nuts and usable spanners). It's an additive heuristic that doesn't account for sequential travel or shared paths, but it's efficiently computable and captures the core requirements.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and computing shortest paths between locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_facts = task.initial # Initial state facts to find all locations

        # 1. Identify all locations
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
        for fact in initial_facts:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc)
        self.locations = list(locations) # Store as list for consistent indexing if needed, though dict is used

        # 2. Build adjacency list for the location graph
        self.graph = {loc: [] for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 4. Identify the nuts that are part of the goal condition
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

    def _bfs(self, start_location):
        """
        Perform BFS from a start location to find shortest paths to all other locations.
        Returns a dictionary mapping location to distance.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            for neighbor in self.graph.get(current_loc, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

        return distances

    def get_distance(self, loc1, loc2):
        """Get the shortest path distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # Should not happen in connected graph, but handle defensively
             return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify the man's current location.
        man_loc = None
        # 3. Identify usable spanners and their locations, and carried spanner
        usable_spanners = set()
        carried_spanner = None
        # Keep track of object locations
        obj_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                obj_locations[obj] = loc
                # Check if this is the man
                if obj.startswith("bob"): # Assuming the man is named 'bob' or similar convention
                     man_loc = loc
            elif match(fact, "usable", "*"):
                 _, spanner = get_parts(fact)
                 usable_spanners.add(spanner)
            elif match(fact, "carrying", "*", "*"):
                 _, man_obj, spanner = get_parts(fact)
                 # Assuming the carried object is a spanner and the carrier is the man
                 carried_spanner = spanner


        # Ensure man_loc is found (should always be the case in valid states)
        if man_loc is None:
             # This state is likely invalid or represents an unsolvable scenario
             return float('inf') # Or handle appropriately

        # 2. Identify loose nuts that are in the goal state
        loose_goal_nuts = set()
        for nut in self.goal_nuts:
            # Check if the nut is currently loose
            if f"(loose {nut})" in state:
                loose_goal_nuts.add(nut)

        # 5. Count the number of loose goal nuts
        n_nuts_to_tighten = len(loose_goal_nuts)

        # 6. Count the number of usable spanners available
        # Usable spanners are those with the (usable ?s) predicate true.
        # We already collected these in usable_spanners set.
        n_usable_spanners = len(usable_spanners)

        # 7. If N_nuts_to_tighten is 0, the heuristic is 0
        if n_nuts_to_tighten == 0:
            return 0

        # 8. If N_nuts_to_tighten > N_usable_spanners, return infinity
        if n_nuts_to_tighten > n_usable_spanners:
             # Problem is unsolvable from this state with available usable spanners
             return float('inf')

        # 9. Calculate heuristic components
        total_cost = 0

        # Cost for tighten_nut actions
        total_cost += n_nuts_to_tighten # Each loose goal nut needs one tighten action

        # Cost for pickup_spanner actions
        # Each nut needs a distinct usable spanner. We need N_nuts_to_tighten spanners.
        # If the man is currently carrying a usable spanner, he needs to pick up N-1 more.
        # If he is not carrying a usable spanner, he needs to pick up N more.
        # However, the heuristic simplifies this: assume N pickups are needed.
        # A slightly better estimate: if he is carrying a *usable* spanner, he might save one pickup.
        # But the spanner becomes unusable after tightening. So he needs a *new* usable spanner for each nut.
        # The simplest is N pickups.
        total_cost += n_nuts_to_tighten # Each loose goal nut needs a spanner, assume pickup needed

        # Travel cost estimate
        travel_cost = 0

        # Travel to each loose goal nut location
        for nut in loose_goal_nuts:
            nut_loc = obj_locations.get(nut)
            if nut_loc and man_loc in self.distances and nut_loc in self.distances[man_loc]:
                 travel_cost += self.get_distance(man_loc, nut_loc)
            else:
                 # Should not happen if locations are consistent and graph connected
                 return float('inf') # Or handle appropriately

        # Travel to each usable spanner location
        # We need N spanners. Which N? The heuristic simplifies by summing travel to *all* usable spanners.
        # This is an overestimation but captures the need to reach spanners.
        for spanner in usable_spanners:
             # Only consider spanners on the ground for pickup travel
             if spanner in obj_locations: # Check if spanner is on the ground
                 spanner_loc = obj_locations[spanner]
                 if man_loc in self.distances and spanner_loc in self.distances[man_loc]:
                      travel_cost += self.get_distance(man_loc, spanner_loc)
                 else:
                      # Should not happen
                      return float('inf') # Or handle appropriately


        total_cost += travel_cost

        return total_cost

