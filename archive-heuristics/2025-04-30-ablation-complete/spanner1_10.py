# Need to import deque for BFS
from collections import deque
# Need to import fnmatch for pattern matching facts
from fnmatch import fnmatch
# Need to import Heuristic base class
# Assuming heuristic_base.py is available in the execution environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base is not found
    # This allows the code to be syntactically correct for testing purposes
    # but it won't run in a real planning environment without the base class.
    print("Warning: heuristics.heuristic_base not found. Using dummy class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found.")


# Helper functions
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
    # Check if the number of parts matches the number of args for a basic match
    if len(parts) != len(args):
        return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all reachable nodes.
    Assumes unweighted graph.
    """
    dist = {start_node: 0}
    queue = deque([start_node])
    while queue:
        u = queue.popleft()
        # Ensure the node exists in the graph keys before accessing neighbors
        if u in graph:
            for v in graph[u]:
                # Ensure neighbor is also a valid graph node key
                if v in graph and v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
    return dist


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all target nuts.
    It sums the fixed costs of necessary actions (tighten, pickup) and an estimated
    travel cost. The travel cost is estimated by greedily visiting the nearest
    required location (either a loose nut location or a usable spanner location
    if spanners are needed).

    # Assumptions
    - There is only one man.
    - Nuts are static at their initial locations.
    - Spanners become permanently unusable after one use.
    - The problem instance is solvable (enough usable spanners exist).
    - The location graph is connected for all relevant locations in solvable instances.

    # Heuristic Initialization
    - Identify the man object, all locations, nuts, and spanners by parsing initial state and static facts.
    - Store the static locations of nuts.
    - Identify the target nuts from the goal conditions.
    - Build the location graph from 'link' facts.
    - Compute all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Determine if the man is currently carrying a usable spanner.
    3. Identify all usable spanners currently on the ground and their locations.
    4. Identify all target nuts from the goal that are still 'loose' in the current state.
    5. If there are no loose target nuts, the heuristic is 0 (goal state).
    6. Calculate the number of loose target nuts (`N_loose`).
    7. Calculate the number of spanners the man needs to pick up from the ground to tighten all `N_loose` nuts, considering if he starts with a usable spanner:
       `spanners_to_pickup_needed = max(0, N_loose - (1 if man is carrying usable spanner else 0))`.
    8. The fixed action cost is `N_loose` (for tighten actions) + `spanners_to_pickup_needed` (for pickup actions).
    9. Calculate the travel cost:
       - Start at the man's current location.
       - Identify the set of locations that need to be visited:
         - The location of every loose target nut.
         - The location of every usable spanner on the ground (if `spanners_to_pickup_needed > 0`).
       - Greedily estimate the travel cost by repeatedly moving to the nearest unvisited location from the set of needed locations until all are visited. Sum the distances of these movements.
    10. The total heuristic value is the fixed action cost plus the estimated travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and computing
        shortest paths.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # --- Parse Objects and Static Facts ---
        self.man_obj = None
        self.all_locations = set()
        self.all_nuts = set()
        self.all_spanners = set()
        self.nut_locations = {} # Map nut object to its location (assumed static)
        self.location_graph = {} # Adjacency list for locations

        # Combine initial state and static facts to find all objects and their types/relations
        all_facts = set(self.initial_state) | set(self.static)

        # First pass to identify object types based on predicate arguments
        potential_types = {}
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                potential_types.setdefault(obj, set()).add('locatable')
                potential_types.setdefault(loc, set()).add('location')
            elif predicate == 'carrying' and len(parts) == 3:
                 man, spanner = parts[1], parts[2]
                 potential_types.setdefault(man, set()).add('man')
                 potential_types.setdefault(spanner, set()).add('spanner')
            elif predicate == 'usable' and len(parts) == 2:
                 spanner = parts[1]
                 potential_types.setdefault(spanner, set()).add('spanner')
            elif predicate == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                potential_types.setdefault(loc1, set()).add('location')
                potential_types.setdefault(loc2, set()).add('location')
            elif predicate in ['tightened', 'loose'] and len(parts) == 2:
                nut = parts[1]
                potential_types.setdefault(nut, set()).add('nut')

        # Assign objects to sets based on inferred types
        for obj, types in potential_types.items():
            if 'man' in types:
                self.man_obj = obj # Assuming only one man
            if 'location' in types:
                self.all_locations.add(obj)
                self.location_graph.setdefault(obj, []) # Ensure all locations are graph nodes
            if 'nut' in types:
                self.all_nuts.add(obj)
            if 'spanner' in types:
                self.all_spanners.add(obj)

        # Second pass to get static nut locations and build graph
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts
             predicate = parts[0]

             if predicate == 'at' and len(parts) == 3 and parts[1] in self.all_nuts:
                  # Store initial nut location, assuming nuts don't move
                  nut, loc = parts[1], parts[2]
                  self.nut_locations[nut] = loc
             elif predicate == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Add links bidirectionally, ensuring locations exist in graph keys
                if loc1 in self.location_graph and loc2 in self.location_graph:
                    self.location_graph[loc1].append(loc2)
                    self.location_graph[loc2].append(loc1)


        # --- Compute All-Pairs Shortest Paths ---
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = bfs(self.location_graph, start_loc)

        # --- Identify Target Nuts ---
        self.target_nuts = set()
        for goal in self.goals:
            # Goal is typically (tightened nut)
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                self.target_nuts.add(nut)

        # Basic check: Ensure man object was found
        if self.man_obj is None:
             # This should not happen in valid spanner instances based on domain predicates
             print("Warning: Man object not found based on predicates. Heuristic might be inaccurate.")


    def get_object_location(self, state, obj):
        """Finds the current location of an object in the state."""
        # Check if the object is at a location
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]
        # If not at a location, check if it's being carried
        # Assuming only man can carry spanners in this domain
        if obj in self.all_spanners and self.man_obj is not None: # Only spanners can be carried by the man
            for fact in state:
                if match(fact, "carrying", self.man_obj, obj):
                     # If carried, its effective location is the man's location
                     return self.get_object_location(state, self.man_obj)
        return None # Object not found at a location or carried

    def get_carried_spanner(self, state):
        """Finds the spanner currently carried by the man, if any."""
        if self.man_obj is None: return None # Cannot carry if no man
        for fact in state:
            if match(fact, "carrying", self.man_obj, "*"):
                return get_parts(fact)[2]
        return None


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Find man's current location
        man_loc = self.get_object_location(state, self.man_obj)
        if man_loc is None:
             # Man must have a location in a valid state. Return infinity if not.
             return float('inf')

        # 2. Determine if man is carrying a usable spanner
        carried_spanner = self.get_carried_spanner(state)
        carrying_usable = carried_spanner is not None and f'(usable {carried_spanner})' in state

        # 3. Identify usable spanners on the ground and their locations
        usable_spanners_on_ground = set()
        usable_spanner_locs = set()
        for spanner in self.all_spanners:
            # Check if spanner is on the ground (i.e., has an 'at' fact) and is usable
            # get_object_location handles carried spanners, so if it returns a location
            # and the spanner is not the one carried, it must be on the ground.
            spanner_loc = self.get_object_location(state, spanner)
            if spanner_loc is not None and spanner != carried_spanner and f'(usable {spanner})' in state:
                 usable_spanners_on_ground.add(spanner)
                 usable_spanner_locs.add(spanner_loc)


        # 4. Identify loose target nuts
        loose_target_nuts = {n for n in self.target_nuts if f'(loose {n})' in state}
        num_loose_nuts = len(loose_target_nuts)

        # 5. Goal state check
        if num_loose_nuts == 0:
            return 0

        # 6. Calculate spanners to pickup needed
        spanners_to_pickup_needed = max(0, num_loose_nuts - (1 if carrying_usable else 0))

        # 8. Fixed action cost (tighten + pickup)
        # Each loose nut needs one tighten action.
        # Man needs N_loose spanners in total. If he starts carrying one usable, he needs N_loose-1 pickups.
        action_cost = num_loose_nuts + spanners_to_pickup_needed

        # 9. Calculate travel cost (Greedy TSP-like)
        travel_cost = 0
        current_loc = man_loc

        # Locations we still need to visit in the greedy path
        # These are locations of loose nuts and locations of usable spanners on the ground (if pickups are needed)
        locations_to_visit = set()

        # Add all loose nut locations
        for nut in loose_target_nuts:
            # Ensure nut location is known (should be from initial state)
            if nut in self.nut_locations:
                locations_to_visit.add(self.nut_locations[nut])
            else:
                 # Should not happen in valid instances
                 print(f"Warning: Location for nut {nut} not found.")
                 return float('inf') # Cannot solve if nut location is unknown

        # Add usable spanner locations if pickups are needed
        if spanners_to_pickup_needed > 0:
             locations_to_visit.update(usable_spanner_locs)

        # If the man starts at a location that is in our set of needed locations,
        # we can consider it "visited" for the travel calculation starting point.
        # This prevents adding travel cost to the initial location if it's relevant.
        if current_loc in locations_to_visit:
             locations_to_visit.remove(current_loc)


        while locations_to_visit:
            nearest_loc = None
            min_dist = float('inf')

            # Find nearest unvisited needed location
            for loc in locations_to_visit:
                 # Ensure distance is computable (location might be isolated)
                 if current_loc in self.dist and loc in self.dist[current_loc]:
                    d = self.dist[current_loc][loc]
                    if d < min_dist:
                        min_dist = d
                        nearest_loc = loc
                 else:
                     # If a needed location is unreachable from the current location,
                     # the problem is likely unsolvable from this state.
                     return float('inf')


            if nearest_loc is None:
                # This should not happen if locations_to_visit is not empty and all are reachable
                # but is a safeguard.
                break

            travel_cost += min_dist
            current_loc = nearest_loc
            locations_to_visit.remove(current_loc) # Mark this location as visited in the greedy path


        # 10. Total heuristic
        total_heuristic = action_cost + travel_cost

        # Heuristic must be 0 at goal. We already handled num_loose_nuts == 0.
        # Ensure heuristic is non-negative. Distances are non-negative, counts are non-negative.

        return total_heuristic
