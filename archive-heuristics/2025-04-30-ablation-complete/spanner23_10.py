from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation for shortest path on the location graph
def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.
    Assumes unweighted graph.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        curr_node = queue.popleft()
        curr_dist = distances[curr_node]

        if curr_node in graph: # Ensure the node exists in the graph keys
            for neighbor in graph[curr_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = curr_dist + 1
                    queue.append(neighbor)

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It sums the estimated cost for each individual loose nut that needs tightening.
    The estimated cost for a single nut includes the tighten action, the pickup action
    for a spanner, and the necessary travel for the man to acquire a spanner and
    reach the nut's location.

    # Assumptions:
    - There is only one man.
    - Spanners are consumed after one use (`tighten_nut` makes them not usable).
    - Nuts stay in their initial locations.
    - The location graph is connected (or relevant parts are).
    - The heuristic calculates the cost for each nut independently and sums them,
      which might overestimate but provides a reasonable estimate for greedy search.
      It assumes that for each nut, the man might need to acquire a new spanner
      and travel, without considering shared travel paths or already-held spanners
      efficiently across multiple nuts in a single trip (beyond the first nut).

    # Heuristic Initialization
    - Parses static facts to build the location graph and precompute shortest path distances
      between all pairs of locations using BFS.
    - Identifies the man's name, spanner names, and nut names from the initial state
      facts to help in identifying objects in subsequent states.
    - Stores the set of goal nuts that need to be tightened.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all nuts that are currently `loose` and are part of the goal (need tightening).
    2. If no such nuts exist, the heuristic is 0 (goal reached for these nuts).
    3. Find the man's current location.
    4. Determine if the man is currently carrying a usable spanner.
    5. Identify all usable spanners that are currently at specific locations (not carried).
    6. Initialize the total heuristic cost to 0.
    7. For each loose nut that needs tightening:
       a. Get the location of this nut.
       b. Calculate the estimated cost to tighten *this specific nut*:
          - Start with a base cost of 1 for the `tighten_nut` action.
          - If the man is currently carrying a usable spanner:
            Add the shortest distance from the man's current location to the nut's location (cost of walking).
          - If the man is not currently carrying a usable spanner:
            - The man needs to acquire a usable spanner. This requires a `pickup_spanner` action (cost 1) plus travel.
            - Find the minimum travel cost to get a usable spanner to the man *at the nut's location*. This involves finding the nearest usable spanner at a location `loc_s`, walking from the man's current location to `loc_s`, picking up the spanner, and walking from `loc_s` to the nut's location. Calculate `dist(man_loc, loc_s) + 1 (pickup) + dist(loc_s, nut_loc)` for all available usable spanners at locations and take the minimum.
            - If no usable spanners are available anywhere (neither carried nor at locations), the problem is unsolvable, and the heuristic should be infinity.
            - Add this minimum acquisition and travel cost to the nut's cost.
       c. Add the estimated cost for this nut to the total heuristic cost.
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and object names."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Build location graph and precompute distances
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1)

        # Add locations mentioned in initial state 'at' facts if not already included
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 locations.add(loc)
                 self.location_graph.setdefault(loc, []) # Ensure location exists in graph even if no links

        self.locations = list(locations) # Store list of all locations
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = bfs(self.location_graph, loc)

        # 2. Identify object names by type (man, spanner, nut)
        # Identify spanners by initial usability, nuts by initial looseness, man by being 'at' a location and not a spanner/nut.
        self.spanner_names = set()
        self.nut_names = set()
        self.man_name = None

        initial_objects_at_loc = {} # {obj: loc}
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                initial_objects_at_loc[obj] = loc

        initial_usable_spanners = {get_parts(fact)[1] for fact in initial_state if match(fact, "usable", "*")}
        initial_loose_nuts = {get_parts(fact)[1] for fact in initial_state if match(fact, "loose", "*")}

        self.spanner_names = initial_usable_spanners
        self.nut_names = initial_loose_nuts

        # Find man: object at a location that is not a spanner or nut
        for obj, loc in initial_objects_at_loc.items():
            if obj not in self.spanner_names and obj not in self.nut_names:
                self.man_name = obj
                break # Assuming one man

        # 3. Store goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts needing tightening
        loose_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}
        nuts_to_tighten = loose_nuts_in_state.intersection(self.goal_nuts)

        if not nuts_to_tighten:
            return 0 # All goal nuts are tightened

        # 2. Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break
        # If man_loc is None, the man is not at a location, which shouldn't happen in valid states.
        # Returning infinity might be appropriate, but let's assume valid states.
        if man_loc is None:
             return float('inf') # Should not happen in valid states

        # 3. Determine if man is carrying a usable spanner
        man_carrying_usable = False
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner_name = get_parts(fact)[2]
                if f"(usable {carried_spanner_name})" in state:
                    man_carrying_usable = True
                break # Assuming one man, one carried item slot (spanner)

        # 4. Identify usable spanners at locations
        usable_spanners_at_loc = {} # {spanner_name: location}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.spanner_names and f"(usable {obj})" in state:
                   usable_spanners_at_loc[obj] = loc

        # 5. Calculate total heuristic cost
        total_cost = 0

        for nut_name in nuts_to_tighten:
            # Find the nut's current location
            nut_loc = None
            for fact in state:
                if match(fact, "at", nut_name, "*"):
                    nut_loc = get_parts(fact)[2]
                    break
            # Assuming nut_loc is always found for a loose nut in a valid state.
            if nut_loc is None:
                 return float('inf') # Should not happen in valid states

            cost_n = 1 # Cost for the tighten_nut action

            if man_carrying_usable:
                # Man has a usable spanner, just needs to walk to the nut
                # Check if distance is known (locations might be disconnected)
                if man_loc not in self.distances or nut_loc not in self.distances[man_loc]:
                     return float('inf') # Unreachable nut location
                cost_n += self.distances[man_loc][nut_loc]
            else:
                # Man needs to acquire a usable spanner first
                min_acq_travel_cost = float('inf')

                # Consider usable spanners at locations
                for s_name, s_loc in usable_spanners_at_loc.items():
                    # Check if distances are known
                    if man_loc not in self.distances or s_loc not in self.distances[man_loc] or \
                       s_loc not in self.distances or nut_loc not in self.distances[s_loc]:
                         continue # This spanner location or path is unreachable

                    acq_travel_cost = self.distances[man_loc][s_loc] + 1 # walk to spanner + pickup
                    acq_travel_cost += self.distances[s_loc][nut_loc] # walk from spanner to nut
                    min_acq_travel_cost = min(min_acq_travel_cost, acq_travel_cost)

                if min_acq_travel_cost == float('inf'):
                    # No usable spanners available anywhere or reachable
                    return float('inf') # Problem unsolvable

                cost_n += min_acq_travel_cost

            total_cost += cost_n

        return total_cost
