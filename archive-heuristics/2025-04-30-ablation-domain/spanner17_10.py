import collections
from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions
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
    # Ensure the number of parts is at least the number of args for zip
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the Heuristic base class if it's not provided externally for standalone testing
# In the target environment, this import will work.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class for testing purposes if the actual one isn't found
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Subclass must implement abstract method")


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose goal nuts. It considers the number of nuts to tighten, the cost
    to acquire a spanner if Bob isn't carrying one, and the travel cost
    for Bob to reach the location of each loose goal nut.

    # Assumptions
    - Nuts are static objects at fixed locations.
    - Spanners are static objects at fixed locations when not carried.
    - Spanners are reusable and do not become unusable after tightening a nut.
    - Links between locations are bidirectional.
    - Bob can only carry one spanner at a time.
    - The goal is to tighten a specific set of nuts.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which nuts need tightening.
    - Parses static facts and initial state to build the location graph based on 'link' predicates and identify all locations.
    - Calculates all-pairs shortest paths between locations using BFS.
    - Identifies all spanners and nuts present in the problem.
    - Stores static nut locations based on the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all nuts that are currently 'loose' and are part of the goal.
    2. If there are no such nuts, the heuristic is 0 (goal reached for nuts).
    3. Find Bob's current location.
    4. Check if Bob is currently carrying a usable spanner.
    5. Initialize the heuristic value:
       - Start with the number of loose goal nuts (representing the minimum 'tighten' actions).
    6. Add the cost to acquire a spanner if Bob is not carrying one:
       - Find all usable spanners and their current locations (if not carried).
       - Calculate the minimum cost for Bob to reach any usable spanner's location and pick it up (shortest distance + 1 for pick action).
       - Add this minimum cost to the heuristic. If no usable spanner is reachable, the state is likely unsolvable, return infinity.
    7. Add the estimated travel cost for Bob to reach each loose goal nut:
       - For each loose goal nut, calculate the shortest distance from Bob's current location to the nut's location.
       - Sum these distances and add to the heuristic. This overestimates travel but guides the search towards states where Bob is closer to the nuts. If any nut location is unreachable, return infinity.
    8. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state to find all objects

        # Identify all spanners and nuts in the problem and collect all locations
        self.all_spanners = set()
        self.all_nuts = set()
        self.all_locations = set()

        # Collect objects and initial locations from the initial state
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 obj, loc = parts[1:]
                 if obj.startswith('spanner'):
                     self.all_spanners.add(obj)
                 elif obj.startswith('nut'):
                     self.all_nuts.add(obj)
                 # Add all locations mentioned in 'at' predicates
                 self.all_locations.add(loc)
             elif parts[0] == 'carrying':
                 carrier, obj = parts[1:]
                 if obj.startswith('spanner'):
                     self.all_spanners.add(obj)

        # Map nut objects to their static locations
        self.nut_locations = {}
        for nut in self.all_nuts:
             # Find the initial location of the nut - assuming they don't move
             # Use next with a default None in case a nut is mentioned but not 'at' a location initially
             nut_loc = next((get_parts(fact)[2] for fact in initial_state if match(fact, "at", nut, "*")), None)
             if nut_loc:
                 self.nut_locations[nut] = nut_loc
                 # Ensure all nut locations are in the set of all locations
                 self.all_locations.add(nut_loc)


        # Build the location graph from 'link' predicates
        self.location_graph = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1)
                # Ensure all linked locations are in the set of all locations
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Ensure all locations from the graph are in the set of all locations
        # (redundant if collected from links, but safe)
        self.all_locations.update(self.location_graph.keys())
        for neighbors in self.location_graph.values():
             self.all_locations.update(neighbors)


        # Calculate all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.all_locations}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Only explore if the current_node is in the graph (has neighbors)
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        # Check if both locations are known and reachable from each other
        # The BFS populates distances for all nodes in self.all_locations.
        # If loc1 or loc2 are not in self.all_locations, they are not valid locations in the graph.
        # If loc2 is not reachable from loc1, distances[loc1][loc2] will be float('inf').
        return self.distances.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose goal nuts and their locations
        nuts_to_tighten = []
        for goal in self.goals:
             # Check if the goal is (tightened ?nut)
             if match(goal, "tightened", "*"):
                 nut = get_parts(goal)[1]
                 # Check if this nut is currently loose in the state
                 if "(loose " + nut + ")" in state:
                     # Find the location of this nut (should be static)
                     if nut in self.nut_locations:
                         loc_n = self.nut_locations[nut]
                         # We assume nuts are static, so if it's loose, it's at its static location
                         # No need to check "(at nut loc_n)" in state again if nut_locations is reliable
                         nuts_to_tighten.append((nut, loc_n))
                     else:
                         # Goal nut not found in static locations - problem with instance?
                         # Treat as unsolvable for this nut.
                         return float('inf')


        # 2. If there are no loose goal nuts, return 0
        if not nuts_to_tighten:
            return 0

        # 3. Find Bob's current location
        bob_loc = next((get_parts(fact)[2] for fact in state if match(fact, "at", "bob", "*")), None)
        if bob_loc is None:
             # Bob's location not found - problem state?
             return float('inf')

        # 4. Check if Bob is currently carrying a usable spanner
        bob_carrying_usable_spanner = False
        for spanner in self.all_spanners:
             if ("(carrying bob " + spanner + ")") in state and ("(usable " + spanner + ")") in state:
                 bob_carrying_usable_spanner = True
                 break

        # 5. Initialize heuristic with tighten actions
        h = len(nuts_to_tighten)

        # 6. Add spanner acquisition cost if needed
        if not bob_carrying_usable_spanner:
            usable_spanner_locs = {}
            for spanner in self.all_spanners:
                 # Check if spanner is usable and at a location (not carried)
                 if ("(usable " + spanner + ")") in state:
                     loc_s = next((get_parts(fact)[2] for fact in state if match(fact, "at", spanner, "*")), None)
                     if loc_s:
                         usable_spanner_locs[spanner] = loc_s

            if not usable_spanner_locs:
                # No usable spanners available anywhere on the ground
                # If Bob isn't carrying one, and none are on the ground, maybe they are unusable or not in the problem?
                # Assuming problem is solvable, this means Bob MUST pick one up from the ground.
                # If none are on the ground, it's unsolvable from this state regarding spanners.
                return float('inf')

            min_spanner_pickup_cost = float('inf')
            for loc_s in usable_spanner_locs.values():
                dist = self.get_distance(bob_loc, loc_s)
                if dist != float('inf'):
                    min_spanner_pickup_cost = min(min_spanner_pickup_cost, dist + 1) # +1 for pick action

            if min_spanner_pickup_cost == float('inf'):
                # Cannot reach any usable spanner on the ground
                return float('inf')

            h += min_spanner_pickup_cost

        # 7. Add estimated travel cost for Bob to reach each nut location
        # This is an additive cost, summing distances from current BobLoc to each nutLoc.
        # This is where non-admissibility comes in, as Bob travels sequentially.
        for nut, loc_n in nuts_to_tighten:
            dist = self.get_distance(bob_loc, loc_n)
            if dist == float('inf'):
                # Cannot reach this nut location
                return float('inf')
            h += dist

        # 8. Return the total calculated heuristic value.
        return h
