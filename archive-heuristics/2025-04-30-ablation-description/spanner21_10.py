from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
# If running in the actual planner environment, remove this dummy class
# and ensure the correct import path is used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Using dummy Heuristic base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass

# --- Helper functions ---
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# --- Heuristic Class ---
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total cost to tighten all loose goal nuts.
    It sums the estimated cost for each loose goal nut independently.
    The estimated cost for a single loose goal nut includes:
    1. The cost of the 'tighten_nut' action (1).
    2. The cost for the man to walk from his current location to the nut's location.
    3. The cost for the man to acquire a usable spanner once he is at the nut's location.

    # Assumptions
    - Nut locations are static throughout the plan.
    - Spanner locations are static until they are picked up.
    - Spanners become unusable after one use of the 'tighten_nut' action.
    - There is exactly one man object in the domain.
    - The location graph formed by 'link' predicates is static and represents bidirectional connections.
    - If the problem is solvable, there are always enough usable spanners available initially.

    # Heuristic Initialization
    - Identify the man object name.
    - Identify all location objects.
    - Build the location graph based on 'link' predicates from static facts.
    - Precompute all-pairs shortest path distances between locations using Breadth-First Search (BFS).
    - Identify all goal nuts and their initial locations (assuming they don't move until tightened).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Find the man's current location.
    2. Determine if the man is currently carrying a spanner and if that spanner is usable.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each goal nut and its location identified during initialization:
        a. Check if the current state contains the fact '(tightened <nut_name>)'.
        b. If the nut is already tightened, its contribution to the heuristic is 0. Continue to the next goal nut.
        c. If the nut is loose (i.e., '(tightened <nut_name>)' is not in the state):
            i. Add 1 to the cost for this nut (representing the 'tighten_nut' action).
            ii. Add the shortest path distance from the man's current location to the nut's location (cost to walk to the nut). Use precomputed distances.
            iii. Calculate the cost to acquire a usable spanner *assuming the man is already at the nut's location*:
                - If the man is currently carrying a usable spanner, this spanner acquisition cost is 0.
                - If the man is carrying a non-usable spanner, add 1 to the spanner acquisition cost (for the 'drop' action).
                - Find all usable spanners that are currently at some location (not carried by the man).
                - If there are no such usable spanners available anywhere, the problem is likely unsolvable from this state; return infinity for the total heuristic.
                - Find the closest available usable spanner to the nut's location (using precomputed distances). The cost to get this spanner is the shortest path distance from the nut's location to the spanner's location plus 1 (for the 'pickup_spanner' action).
                - Add this minimum pickup cost to the spanner acquisition cost.
            iv. Add the calculated spanner acquisition cost to the cost for this nut.
            v. Add the total cost for this nut to the overall total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building the location graph."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        self.man_name = None
        self.locations = set()
        self.goal_nuts_locations = {}

        # Collect locations from static links
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                self.locations.add(get_parts(fact)[1])
                self.locations.add(get_parts(fact)[2])

        # Collect locations and identify man from initial state
        spanner_names_in_init = set()
        nut_names_in_init = set()
        locatable_objects_in_init = set() # Objects that appear in 'at' facts

        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1:]
                locatable_objects_in_init.add(obj)
                self.locations.add(loc)
            elif parts[0] == 'carrying' and len(parts) == 3:
                 self.man_name = parts[1] # Man is the first arg of 'carrying'
                 spanner_names_in_init.add(parts[2])
            elif parts[0] == 'usable' and len(parts) == 2:
                 spanner_names_in_init.add(parts[1])
            elif parts[0] == 'loose' and len(parts) == 2:
                 nut_names_in_init.add(parts[1])
            elif parts[0] == 'tightened' and len(parts) == 2:
                 nut_names_in_init.add(parts[1])

        # If man not found via 'carrying', assume it's the locatable object that isn't a spanner or nut
        if self.man_name is None:
             for obj in locatable_objects_in_init:
                 if obj not in spanner_names_in_init and obj not in nut_names_in_init:
                     self.man_name = obj
                     break # Assuming only one man

        self.locations = list(self.locations) # Convert set to list

        # Build graph and compute distances
        self.location_graph = {loc: [] for loc in self.locations}
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                # Ensure locations are in our collected set before adding to graph
                if l1 in self.location_graph and l2 in self.location_graph:
                     self.location_graph[l1].append(l2)
                     self.location_graph[l2].append(l1) # Assuming links are bidirectional

        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = self.bfs(loc, self.location_graph)

        # Store goal nuts and their initial locations
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut_name = get_parts(goal)[1]
                # Find the initial location of this nut
                nut_location = None
                for fact in self.initial_state:
                    if match(fact, "at", nut_name, "*"):
                        nut_location = get_parts(fact)[2]
                        break
                if nut_location:
                    self.goal_nuts_locations[nut_name] = nut_location
                else:
                    # Goal nut location not found in initial state - indicates a problem
                    # This heuristic might return inf or behave unexpectedly if a goal nut isn't placed.
                    # For now, we assume valid instances where goal nuts are initially located.
                    pass


    def bfs(self, start_node, graph):
        """Computes shortest path distances from start_node to all other nodes in the graph."""
        distances = {node: float('inf') for node in graph}
        if start_node in distances: # Ensure start_node is in the graph nodes
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current = queue.popleft()
                # Check if current node exists in graph adjacency list (important if graph was built partially)
                if current not in graph:
                     continue

                for neighbor in graph[current]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Retrieves the precomputed shortest distance between two locations."""
        # Check if locations exist and distance was computed
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
            # This indicates an issue (e.g., location not in graph, disconnected graph)
            # Returning infinity implies this path is impossible or very costly
            return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Find the man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None:
             # Man's location is unknown - problem state is weird or unsolvable
             return float('inf')

        # 2. Determine if the man is currently carrying a spanner and if it's usable
        carried_spanner = None
        is_carried_usable = False
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]
                # Check if the carried spanner is usable in the current state
                if f"(usable {carried_spanner})" in state:
                    is_carried_usable = True
                break # Assuming man carries at most one spanner

        total_heuristic_cost = 0

        # 4. Iterate through each goal nut
        for nut_name, nut_location in self.goal_nuts_locations.items():
            # a. Check if the nut is already tightened
            if f"(tightened {nut_name})" in state:
                continue # Nut is already tightened, cost is 0 for this nut

            # c. If the nut is loose:
            # i. Add 1 for the 'tighten_nut' action
            cost_for_this_nut = 1

            # ii. Add cost to walk to the nut
            walk_to_nut_cost = self.get_distance(man_location, nut_location)
            if walk_to_nut_cost == float('inf'):
                 # Cannot reach the nut location from man's current location
                 return float('inf')
            cost_for_this_nut += walk_to_nut_cost

            # iii. Calculate cost to acquire a usable spanner at the nut's location
            spanner_acquisition_cost = 0
            if not is_carried_usable:
                # Need to get a usable spanner
                drop_cost = 1 if carried_spanner else 0 # Cost to drop current spanner if any
                spanner_acquisition_cost += drop_cost

                min_pickup_cost_from_nut_loc = float('inf')
                available_usable_spanners = []

                # Find usable spanners that are currently at some location (not carried)
                for fact in state:
                    if match(fact, "usable", "*"):
                        s_name = get_parts(fact)[1]
                        # Check if this spanner is not the one the man is carrying (if any)
                        # and is currently at a location (not carried)
                        is_at_location = any(match(loc_fact, "at", s_name, "*") for loc_fact in state)
                        is_carried_by_man = (carried_spanner and s_name == carried_spanner)

                        if is_at_location and not is_carried_by_man:
                             # Find its location
                             for loc_fact in state:
                                 if match(loc_fact, "at", s_name, "*"):
                                     s_loc = get_parts(loc_fact)[2]
                                     available_usable_spanners.append((s_name, s_loc))
                                     break # Found location, move to next spanner

                if not available_usable_spanners:
                     # No usable spanners available anywhere - problem likely unsolvable
                     return float('inf')

                # Find the closest available usable spanner to the nut's location
                for s_name, s_loc in available_usable_spanners:
                     walk_from_nut_to_spanner_cost = self.get_distance(nut_location, s_loc)
                     if walk_from_nut_to_spanner_cost == float('inf'):
                          # Cannot reach this spanner from the nut location
                          continue # Try next spanner

                     cost = walk_from_nut_to_spanner_cost + 1 # Walk from nut loc + pickup
                     min_pickup_cost_from_nut_loc = min(min_pickup_cost_from_nut_loc, cost)

                if min_pickup_cost_from_nut_loc == float('inf'):
                     # Could not find a reachable usable spanner from the nut location
                     return float('inf')

                spanner_acquisition_cost += min_pickup_cost_from_nut_loc

            cost_for_this_nut += spanner_acquisition_cost

            # v. Add the cost for this nut to the total
            total_heuristic_cost += cost_for_this_nut

        # 5. Return the total heuristic cost
        return total_heuristic_cost
