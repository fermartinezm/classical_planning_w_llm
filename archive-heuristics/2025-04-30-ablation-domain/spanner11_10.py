from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have the same number of components as the pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the cost to acquire a spanner (if needed), the travel cost to reach
    the locations of the loose nuts, and the cost of the tighten actions themselves.

    # Assumptions
    - There is only one agent, 'bob'.
    - Nuts need to be tightened using a usable spanner.
    - Bob must be at the same location as the nut and carrying a usable spanner to tighten it.
    - Locations are connected by bidirectional links.
    - All locations, spanners, and nuts involved in the problem are reachable from Bob's initial location
      or become reachable through movement. (The heuristic handles unreachable cases by returning infinity).

    # Heuristic Initialization
    - Extracts all unique location names from the initial state and static facts involved in 'at' or 'link' predicates.
    - Builds a graph of locations based on 'link' predicates.
    - Computes all-pairs shortest paths between locations using BFS. Stores distances in a dictionary.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Goal Progress:** Find all nuts that are currently loose. If there are no loose nuts, the goal is achieved, and the heuristic is 0.
    2.  **Base Cost (Tightening):** Add the number of loose nuts to the heuristic. Each loose nut requires at least one 'tighten' action.
    3.  **Identify Bob's State:** Determine Bob's current location and whether he is carrying a spanner.
    4.  **Identify Required Tools:** Find all usable spanners and their current locations.
    5.  **Cost to Get Spanner-Ready:**
        *   If Bob is already carrying a spanner, the cost to become spanner-ready is 0. The starting point for subsequent travel is Bob's current location.
        *   If Bob is not carrying a spanner, he must acquire one. Find the usable spanner that is nearest to Bob's current location (in terms of shortest path distance). The cost to become spanner-ready is the distance to this nearest spanner plus 1 (for the 'pickup' action). If no usable spanner is reachable, the problem is unsolvable from this state, and the heuristic is infinity. The starting point for subsequent travel is the location of this nearest spanner.
        *   Add this calculated cost to the total heuristic.
    6.  **Cost to Reach Work Areas:**
        *   Identify all unique locations where the loose nuts are situated.
        *   Calculate the maximum shortest path distance from the spanner-ready starting point (determined in step 5) to any of these loose nut locations. This estimates the minimum travel needed to get "close" to all work sites.
        *   If any loose nut location is unreachable from the spanner-ready starting point, the problem is unsolvable, and the heuristic is infinity.
        *   Add this maximum distance to the total heuristic.
    7.  **Total Heuristic:** The sum accumulated in steps 2, 5, and 6 is the estimated number of actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing shortest paths.
        """
        # Store task details if needed, though this heuristic primarily uses static info and state
        # self.goals = task.goals
        # self.initial_state = task.initial_state

        # 1. Collect all unique location names from initial state and static facts
        locations = set()
        # Collect from static facts (links define locations)
        for fact in task.static:
             parts = get_parts(fact)
             if parts and parts[0] == 'link' and len(parts) == 3:
                 locations.add(parts[1])
                 locations.add(parts[2])
        # Collect from initial state facts (objects are at locations)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 # Assume the third argument of 'at' is always a location
                 locations.add(parts[2])

        self.locations = list(locations) # Convert to list if order matters, otherwise set is fine

        # 2. Build the location graph (adjacency list)
        self.graph = {loc: [] for loc in self.locations}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations are in our collected set before adding links
                if loc1 in self.graph and loc2 in self.graph:
                    self.graph[loc1].append(loc2)
                    self.graph[loc2].append(loc1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.dist[start_node][start_node] = 0

            while q:
                curr, d = q.popleft()

                # Ensure current node is in the graph (handle collected locations with no links)
                if curr not in self.graph:
                    continue

                for neighbor in self.graph.get(curr, []): # Use .get for safety
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start_node][neighbor] = d + 1
                        q.append((neighbor, d + 1))

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance, returning infinity if unreachable or location unknown."""
        # Check if locations are known
        if loc1 not in self.locations or loc2 not in self.locations:
             return float('inf')
        # Return precomputed distance, default to infinity if no path was found by BFS
        return self.dist.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant information from the current state
        loose_nuts = set()
        obj_locations = {}
        bob_location = None
        bob_carrying_spanner = False
        usable_spanners = set() # Names of usable spanners

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            if parts[0] == 'loose' and len(parts) == 2:
                loose_nuts.add(parts[1])
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                obj_locations[obj] = loc
                if obj == 'bob':
                    bob_location = loc
            elif parts[0] == 'carrying' and len(parts) == 3 and parts[1] == 'bob':
                 # We just need to know *if* he's carrying *a* spanner
                 bob_carrying_spanner = True
            elif parts[0] == 'usable' and len(parts) == 2:
                 usable_spanners.add(parts[1])

        # Ensure Bob's location is known
        if bob_location is None:
             # Bob's location is unknown, likely an invalid state or domain issue
             return float('inf')

        # Filter loose nuts to only those currently 'at' a known location
        loose_nuts_current = {nut for nut in loose_nuts if nut in obj_locations}
        nut_locations = {nut: obj_locations[nut] for nut in loose_nuts_current}

        # If no loose nuts, goal is reached
        if not loose_nuts_current:
            return 0

        # 2. Base Cost (Tightening): Add cost for each tighten action
        h = len(loose_nuts_current)

        # 3. Bob's state already found

        # 4. Usable spanners already found

        # 5. Calculate cost to get spanner-ready
        C_spanner_needed = 0
        L_start_travel = bob_location # Default start location for travel

        if not bob_carrying_spanner:
            # Find locations of usable spanners that are currently 'at' a location
            usable_spanner_locs = {obj_locations[s] for s in usable_spanners if s in obj_locations}

            if not usable_spanner_locs:
                # No usable spanners available/reachable in the current state
                return float('inf')

            min_dist_to_spanner = float('inf')
            L_spanner_nearest = None

            # Find the nearest reachable usable spanner location from Bob
            for loc in usable_spanner_locs:
                dist = self.get_distance(bob_location, loc)
                if dist < min_dist_to_spanner:
                    min_dist_to_spanner = dist
                    L_spanner_nearest = loc

            if L_spanner_nearest is None or min_dist_to_spanner == float('inf'):
                 # No reachable usable spanner found
                 return float('inf')

            C_spanner_needed = min_dist_to_spanner + 1 # move to spanner + pickup action
            L_start_travel = L_spanner_nearest # Travel to nuts starts from where spanner is picked up

        h += C_spanner_needed

        # 6. Calculate travel cost to reach loose nuts
        loose_nut_locations = set(nut_locations.values())

        max_dist_to_nut = 0
        for loc in loose_nut_locations:
            dist = self.get_distance(L_start_travel, loc)
            if dist == float('inf'):
                # A loose nut is in an unreachable location from the spanner-ready point
                return float('inf')
            max_dist_to_nut = max(max_dist_to_nut, dist)

        h += max_dist_to_nut

        # 7. Return total heuristic
        return h

