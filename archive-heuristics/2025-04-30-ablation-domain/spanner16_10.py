# Import necessary modules
from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base is available in a 'heuristics' directory
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected input formats
        # print(f"Warning: get_parts received non-fact string: {fact}")
        return [] # Return empty list for invalid facts

    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must be at least the number of non-wildcard arguments
    if len(parts) < len([arg for arg in args if arg != '*']):
         return False

    # Check if each part matches the corresponding arg pattern
    # zip stops at the shortest sequence, which is appropriate here.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It sums the estimated cost for each individual loose nut that is a goal. The cost
    for a nut includes moving Bob to the nut's location with a usable spanner and
    performing the tighten action.

    # Assumptions
    - The goal is to tighten all nuts specified in the task's goal conditions.
    - Bob is the only agent who can perform actions.
    - Bob can carry multiple spanners.
    - The location graph defined by `link` predicates is undirected.
    - The `usable` status of a spanner does not change during planning. Usable spanners are identified from the initial state.
    - Nuts are always at a location (not in containers).
    - The 'shed' and 'gate' locations are part of the linked graph if they appear in 'link' facts. If they appear only in 'at' facts but not 'link', they are isolated unless connected via other locations. The BFS handles this by only exploring linked locations. If a nut or spanner is at an isolated location, distance will be infinity.

    # Heuristic Initialization
    - Build the location graph from `link` predicates found in static facts.
    - Compute shortest path distances between all pairs of locations using BFS.
    - Identify all usable spanners from the initial state.
    - Identify all nuts that are goals (need tightening) from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify Bob's current location.
    2. Identify which usable spanners Bob is currently carrying.
    3. Identify the current location of all usable spanners not carried by Bob.
    4. Initialize total heuristic cost `h = 0`.
    5. Iterate through each nut that is a goal (i.e., needs to be tightened).
    6. For a goal nut `N`:
       - Check if `(tightened N)` is true in the current state. If yes, the nut is done; continue to the next nut.
       - If `(tightened N)` is false, this nut needs tightening.
       - Find the location of nut `N` (`LocN`).
       - The cost for this nut is at least 1 (for the `tighten` action itself). Add 1 to a temporary nut cost.
       - **Cost to get Bob to `LocN` with a usable spanner:**
         - Check if Bob is currently carrying *any* usable spanner.
         - If Bob is carrying a usable spanner:
           - The cost to get Bob to `LocN` is the shortest distance from Bob's current location (`LocBob`) to `LocN`. Add this distance to the temporary nut cost. If the location is unreachable, the distance is infinity, making the nut cost infinity.
         - If Bob is NOT carrying a usable spanner:
           - Bob needs to get a usable spanner first.
           - Find the minimum cost to get Bob to a location `LocY` where a usable spanner `spannerY` is available (`(at spannerY LocY)`), pick it up, and then move to `LocN`.
           - The cost for picking up is 1 (`pickup` action).
           - The cost for moving is distance(LocBob, LocY) + distance(LocY, LocN).
           - The total movement+pickup cost for a spanner at `LocY` is distance(LocBob, LocY) + 1 + distance(LocY, LocN).
           - Find the minimum such cost over all usable spanners not currently carried by Bob that are currently 'at' a location.
           - Add this minimum cost to the temporary nut cost. If no such spanner exists or is reachable, the cost is infinite.
       - Add the temporary nut cost to the total heuristic `h`.
    7. Return `h`.

    Note: This sums costs independently per nut. It doesn't account for Bob tightening multiple nuts on one trip or using the same spanner for multiple nuts. This is a relaxation.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building the location graph."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Need initial state to find usable spanners

        # Build the location graph from 'link' predicates
        self.location_graph = {}
        locations = set()
        # Collect all locations mentioned in 'link' facts
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure correct number of parts for a link fact
                    _, loc1, loc2 = parts
                    locations.add(loc1)
                    locations.add(loc2)
                    self.location_graph.setdefault(loc1, set()).add(loc2)
                    self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        self.locations = list(locations) # Store locations as a list

        # Compute all-pairs shortest paths using BFS from each location in the graph
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify all usable spanners (they are static according to assumption)
        # We find them in the initial state
        self.usable_spanners = {get_parts(fact)[1] for fact in initial_state if match(fact, "usable", "*")}

        # Identify nuts that are goals (need tightening)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def _bfs(self, start_location):
        """Performs BFS from a start location to find distances to all other locations reachable within the graph."""
        distances = {loc: float('inf') for loc in self.locations}
        if start_location in distances: # Only start BFS if the location is part of the graph
            distances[start_location] = 0
            queue = deque([start_location])

            while queue:
                current_loc = queue.popleft()
                current_dist = distances[current_loc]

                if current_loc in self.location_graph: # Check if current_loc has neighbors in the graph
                    for neighbor in self.location_graph[current_loc]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations in the graph."""
        # If either location is not in the graph, or loc2 is unreachable from loc1, distance is inf
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Pre-parse state facts for quicker access
        state_facts_map = {}
        for fact in state:
             parts = get_parts(fact)
             if parts:
                 predicate = parts[0]
                 args = tuple(parts[1:])
                 state_facts_map.setdefault(predicate, []).append(args)

        # Find Bob's location
        bob_location = None
        for args in state_facts_map.get("at", []):
            if len(args) == 2 and args[0] == 'bob':
                bob_location = args[1]
                break

        if bob_location is None:
             # Bob must always be at a location in a valid state.
             # If not found, the state is inconsistent or unsolvable.
             return float('inf') # Indicate unsolvable/invalid state

        # Find usable spanners Bob is carrying
        carrying_usable_spanners = set()
        for args in state_facts_map.get("carrying", []):
            if len(args) == 2 and args[0] == 'bob':
                 spanner = args[1]
                 if spanner in self.usable_spanners: # Check if the carried spanner is usable
                     carrying_usable_spanners.add(spanner)

        bob_has_usable_spanner = len(carrying_usable_spanners) > 0

        # Find locations of usable spanners not carried by Bob
        available_spanner_locations = {} # Map spanner -> location
        for spanner in self.usable_spanners:
            if spanner not in carrying_usable_spanners:
                 # Find where this spanner is
                 for args in state_facts_map.get("at", []):
                     if len(args) == 2 and args[0] == spanner:
                         available_spanner_locations[spanner] = args[1]
                         break # Found location, move to next spanner


        total_cost = 0  # Initialize action cost counter.

        # Iterate through each nut that needs tightening (is a goal nut)
        for nut in self.goal_nuts:
            # Check if the nut is already tightened in the current state
            if ("tightened", nut) in state_facts_map.get("tightened", []):
                continue # This nut is done

            # Find the location of the loose nut
            nut_location = None
            for args in state_facts_map.get("at", []):
                if len(args) == 2 and args[0] == nut:
                    nut_location = args[1]
                    break

            if nut_location is None:
                 # A goal nut that is not tightened must be loose and at a location.
                 # If not found, the state is inconsistent or unsolvable.
                 return float('inf') # Indicate unsolvable/invalid state

            # Cost for this nut: 1 (tighten action) + cost to get Bob to nut_location with spanner
            nut_cost = 1 # Cost of the tighten action

            # Cost to get Bob to nut_location with a spanner
            if bob_has_usable_spanner:
                # Bob already has a usable spanner, just need to move him
                move_cost = self.get_distance(bob_location, nut_location)
                if move_cost == float('inf'):
                    # Nut location is unreachable from Bob's current location
                    return float('inf')
                nut_cost += move_cost
            else:
                # Bob needs to get a usable spanner first
                min_spanner_acquisition_cost = float('inf')

                # Consider picking up an available usable spanner
                for spanner, spanner_loc in available_spanner_locations.items():
                    # Cost to get this spanner and then go to the nut:
                    # Move Bob to spanner_loc + Pickup spanner + Move Bob (with spanner) to nut_location
                    dist_to_spanner = self.get_distance(bob_location, spanner_loc)
                    dist_spanner_to_nut = self.get_distance(spanner_loc, nut_location)

                    if dist_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                         cost = dist_to_spanner + 1 + dist_spanner_to_nut
                         min_spanner_acquisition_cost = min(min_spanner_acquisition_cost, cost)

                if min_spanner_acquisition_cost == float('inf'):
                     # No reachable usable spanner available on the ground, and Bob isn't carrying one.
                     # This goal nut is unreachable.
                     return float('inf')

                nut_cost += min_spanner_acquisition_cost

            # Add the cost for this nut to the total
            total_cost += nut_cost

        return total_cost
