# Import necessary modules
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing whitespace
    fact_str = str(fact).strip()
    # Remove parentheses and split by whitespace
    # Handle cases with empty facts or malformed strings defensively
    if not fact_str or fact_str[0] != '(' or fact_str[-1] != ')':
        return []
    return fact_str[1:-1].split()

# Helper function to match a PDDL fact against a pattern
# This helper is not strictly used in the final heuristic logic but kept as per example structure.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if not parts: # Handle empty parts from get_parts
        return False
    # Check if the number of parts matches the number of args, or if args contains wildcards allowing fewer parts
    if len(parts) != len(args) and '*' not in args:
        return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the number of tightening actions needed, the number of spanners
    that need to be picked up, and an estimate of the travel cost to reach
    relevant locations (nuts and spanners).

    # Assumptions
    - Links between locations are bidirectional.
    - Spanners are consumed (become unusable) after one tightening action.
    - The problem instance is solvable (enough usable spanners exist in total).
    - The man can carry multiple spanners simultaneously.
    - The man's name can be identified from the initial state (e.g., by looking
      for the object in an 'at' fact that is not a nut or spanner, or the first
      argument of a 'carrying' fact).

    # Heuristic Initialization
    - Parses static facts to build a graph of locations based on 'link' predicates.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies all unique locations present in the domain/problem.
    - Identifies the name of the man object from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to identify:
       - The man's current location.
       - All loose nuts and their locations.
       - All usable spanners (both carried and on the ground) and their locations.
       - All spanners currently carried by the man.
    2. Calculate the number of loose nuts (`N_loose`). If `N_loose` is 0, the heuristic is 0 (goal state).
    3. Calculate the number of usable spanners the man is currently carrying (`N_usable_carried`).
    4. Calculate the number of additional usable spanners needed from the ground: `N_needed_from_ground = max(0, N_loose - N_usable_carried)`.
    5. Initialize the heuristic value: `h = N_loose` (for the tighten actions) `+ N_needed_from_ground` (for the pickup actions).
    6. Identify the set of 'relevant locations' the man needs to visit. This set includes:
       - The locations of all loose nuts.
       - If `N_needed_from_ground > 0`, it also includes the locations of all usable spanners that are currently on the ground.
    7. If the set of relevant locations is not empty (which it won't be if `N_loose > 0`):
       - Find the minimum shortest distance from the man's current location to any location in the set of relevant locations (`min_dist_to_first_stop`).
       - Estimate the travel cost as `min_dist_to_first_stop + max(0, len(relevant_locations_to_visit) - 1)`. This accounts for reaching the first relevant location and then potentially moving between the remaining distinct relevant locations.
       - Add this estimated travel cost to the heuristic value `h`.
    8. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing
        shortest paths, and identifying the man's name.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find man and all locations

        # Build location graph from link facts
        locations = set()
        links = []
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                links.append((loc1, loc2))
                locations.add(loc1)
                locations.add(loc2)

        # Add locations mentioned in the initial state (objects' locations)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at':
                 loc = parts[2]
                 locations.add(loc)

        self.locations = list(locations) # Use a list for consistent ordering if needed, though set is fine for lookup
        self.graph = {loc: [] for loc in self.locations}
        for loc1, loc2 in links:
            # Ensure locations exist in the graph dictionary before appending
            if loc1 in self.graph and loc2 in self.graph:
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Links are bidirectional
            # else: # Should not happen if locations set is built correctly
            #    print(f"Warning: Link {loc1}-{loc2} involves unknown location.")


        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

        # Identify the man's name from the initial state
        self.man_name = None
        # Try finding the object that is the first argument of a 'carrying' fact
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'carrying':
                self.man_name = parts[1]
                break
        # If not found, try finding an object in an 'at' fact that is not a nut or spanner
        if self.man_name is None:
             for fact in initial_state:
                parts = get_parts(fact)
                if parts and parts[0] == 'at':
                    obj_name = parts[1]
                    # Crude check: assume nuts start with 'nut', spanners with 'spanner'
                    if not obj_name.startswith('nut') and not obj_name.startswith('spanner'):
                         self.man_name = obj_name
                         break
        # Fallback if man's name couldn't be determined (e.g., problem uses different naming)
        if self.man_name is None:
             # print("Warning: Could not determine man's name from initial state. Assuming 'bob'.")
             self.man_name = 'bob' # Assume 'bob' based on the example state

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        # Ensure start_node is in the locations list/set
        if start_node not in self.locations:
             # print(f"Warning: BFS started from unknown location {start_node}.")
             return distances # Return all infinities if start node is invalid

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            curr = queue.popleft()

            # Check if curr is a valid key in the graph
            if curr not in self.graph:
                 # This might happen if a location from initial state was added but has no links
                 continue

            for neighbor in self.graph.get(curr, []): # Use .get for safety
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Parse the current state
        man_loc = None
        nut_locations = {} # {nut_name: location}
        spanner_locations = {} # {spanner_name: location} # Location of spanners on the ground
        carried_spanners = set() # {spanner_name}
        usable_spanners = set() # {spanner_name}
        loose_nuts = set() # {nut_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue

            predicate = parts[0]
            if predicate == 'at':
                obj_name, loc_name = parts[1], parts[2]
                if obj_name == self.man_name:
                    man_loc = loc_name
                elif obj_name.startswith('nut'):
                    nut_locations[obj_name] = loc_name
                elif obj_name.startswith('spanner'):
                    spanner_locations[obj_name] = loc_name # This is location on the ground
            elif predicate == 'carrying':
                # Assuming the man is always the first argument
                s_name = parts[2]
                carried_spanners.add(s_name)
            elif predicate == 'usable':
                s_name = parts[1]
                usable_spanners.add(s_name)
            elif predicate == 'loose':
                n_name = parts[1]
                loose_nuts.add(n_name)
            # We don't need 'tightened' facts explicitly, as 'loose' being absent implies tightened.
            # We also don't need 'link' facts as they are static.

        # If man_loc is not found, something is wrong with the state or parsing
        if man_loc is None:
             # This shouldn't happen in a valid state, but handle defensively
             # print("Error: Man's location not found in state.")
             return float('inf') # Should not occur in solvable problems

        # Calculate key quantities
        N_loose = len(loose_nuts)

        # If no loose nuts, goal is reached
        if N_loose == 0:
            return 0

        # Find usable spanners the man is carrying
        usable_carried_spanners = carried_spanners.intersection(usable_spanners)
        N_usable_carried = len(usable_carried_spanners)

        # Find usable spanners on the ground
        usable_ground_spanners = {s for s in usable_spanners if s not in carried_spanners}

        # Calculate additional usable spanners needed from the ground
        N_needed_from_ground = max(0, N_loose - N_usable_carried)

        # --- Heuristic Calculation ---
        # Base cost: 1 action for each tighten_nut and 1 action for each pickup_spanner needed
        h = N_loose + N_needed_from_ground

        # Identify relevant locations to visit
        relevant_locations_to_visit = set()

        # Add locations of loose nuts
        for nut_name in loose_nuts:
             if nut_name in nut_locations: # Ensure location is known
                 relevant_locations_to_visit.add(nut_locations[nut_name])
             # else: # Should not happen in valid states
             #    print(f"Warning: Location for loose nut {nut_name} not found in state.")


        # If we need spanners from the ground, add locations of usable ground spanners
        if N_needed_from_ground > 0:
            for spanner_name in usable_ground_spanners:
                 if spanner_name in spanner_locations: # Ensure location is known (spanner is on the ground)
                     relevant_locations_to_visit.add(spanner_locations[spanner_name])
                 # else: # Should not happen in valid states
                 #    print(f"Warning: Location for usable ground spanner {spanner_name} not found in state.")


        # Estimate travel cost
        travel_cost = 0
        if relevant_locations_to_visit:
            # Distance to the first relevant location
            min_dist_to_first_stop = float('inf')

            # Ensure man_loc is a valid start node for distance lookup
            if man_loc in self.dist:
                for target_loc in relevant_locations_to_visit:
                    # Ensure target_loc is a valid destination node
                    if target_loc in self.dist[man_loc]:
                         min_dist_to_first_stop = min(min_dist_to_first_stop, self.dist[man_loc][target_loc])
                    # else: # Should not happen if locations set is built correctly
                    #    print(f"Warning: Target location {target_loc} not found in distances from {man_loc}.")
            # else: # Should not happen if man_loc was added to locations in __init__
            #    print(f"Warning: Man's location {man_loc} not found in precomputed distances.")


            # If any relevant location is reachable
            if min_dist_to_first_stop != float('inf'):
                 # Cost to reach the first location + 1 action for each additional distinct location to visit
                 travel_cost = min_dist_to_first_stop + max(0, len(relevant_locations_to_visit) - 1)
            else:
                 # This state is likely unsolvable if there are loose nuts but no reachable relevant locations
                 # print("Error: Relevant locations are unreachable from man's location.")
                 return float('inf') # Problem is likely unsolvable from this state

        # Add travel cost to the heuristic
        h += travel_cost

        return h
