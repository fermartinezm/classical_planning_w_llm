from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Removes surrounding brackets and splits by space."""
    # Handle potential empty fact string or malformed string gracefully
    if not fact_string or fact_string[0] != '(' or fact_string[-1] != ')':
         # Depending on expected input robustness, could log a warning or raise error
         # For a heuristic, returning None might be safer than crashing
         return None
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function for BFS
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: math.inf for node in graph}
    if start_node in graph: # Ensure start_node is in the graph keys
        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            # Check if current_node is still valid in graph (should be if added from graph keys)
            if current_node in graph:
                for neighbor in graph.get(current_node, []):
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

# Helper function to compute all-pairs shortest paths
def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes in the graph."""
    all_paths = {}
    for start_node in graph:
        all_paths[start_node] = bfs(graph, start_node)
    return all_paths

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    Estimates the cost to reach the goal by summing the number of untightened
    goal nuts (representing tighten actions), the number of usable spanners
    that need to be picked up from the ground, and the minimum travel cost
    to reach any of the locations where a required action (tightening a nut
    or picking up a needed spanner) can take place.

    Assumptions:
    - The PDDL domain structure is as provided (spanner domain).
    - Object types (man, spanner, nut, location) can be inferred from predicate
      signatures.
    - Links between locations are bidirectional.
    - The goal is to tighten a specific set of nuts.
    - Spanners become unusable after one use for tightening.
    - No actions exist to make spanners usable again or move nuts/spanners
      except by the man picking up a spanner.
    - There is exactly one man object in the domain.

    Heuristic Initialization:
    - Parses all facts and operator effects/preconditions from the task
      definition to identify all objects and their types (man, spanner,
      nut, location).
    - Builds a graph of locations based on 'link' predicates found in static facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Stores the static locations of nuts from the initial state.
    - Stores the set of goal nuts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract the man's current location from the state.
    2. Identify all loose nuts that are part of the goal from the state.
    3. Identify all usable spanners currently carried by the man from the state.
    4. Identify all usable spanners currently on the ground from the state.
    5. Count the number of loose goal nuts (`k_loose`).
    6. Count the number of usable spanners carried (`k_carried`).
    7. Count the number of usable spanners on the ground (`k_ground`).
    8. If `k_loose` is 0, the goal is reached, return 0.
    9. If `k_loose` is greater than the total number of available usable spanners (`k_carried + k_ground`), the goal is unreachable, return infinity.
    10. Calculate the number of additional usable spanners needed from the ground: `needed_pickups = max(0, k_loose - k_carried)`.
    11. Identify the set of locations where nuts need tightening (`Nut_locs`). These locations are static and precomputed.
    12. Identify the set of locations where usable spanners are available on the ground (`Spanner_locs`). These locations are extracted from the current state.
    13. Determine the set of locations where required actions can take place (`Required_locs`). This includes all `Nut_locs` and the locations of the `needed_pickups` usable spanners from `Spanner_locs` that are closest to the man's current location.
    14. Calculate the minimum travel cost from the man's current location to any location in `Required_locs`. If `Required_locs` is empty (should only happen if `k_loose` is 0, handled in step 8), travel cost is 0. If any required location is unreachable from the man's current location, travel cost is infinity.
    15. The heuristic value is the sum of `k_loose` (estimated tighten actions), `needed_pickups` (estimated pickup actions), and the calculated minimum travel cost.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task # Store task for access to initial_state, goals, static, operators

        # --- Initialization: Parse domain structure and precompute ---

        self.locations = set()
        self.nuts = set()
        self.spanners = set()
        self.men = set()
        self.goal_nuts = set()
        self.nut_locations = {} # nut -> location (static)

        # Collect all relevant facts from initial state, goals, static, and operators
        all_facts_and_effects = set(task.initial_state) | task.goals | task.static
        for op in task.operators:
            all_facts_and_effects |= op.preconditions | op.add_effects | op.del_effects

        # Infer object types based on predicate signatures
        for fact_str in all_facts_and_effects:
            parts = parse_fact(fact_str)
            if parts is None or len(parts) == 0: continue # Skip malformed facts

            pred = parts[0]
            if pred == 'at':
                # (at ?m - locatable ?l - location)
                # parts[1] is locatable, parts[2] is location
                self.locations.add(parts[2])
            elif pred == 'carrying':
                # (carrying ?m - man ?s - spanner)
                self.men.add(parts[1])
                self.spanners.add(parts[2])
            elif pred == 'usable':
                # (usable ?s - spanner)
                self.spanners.add(parts[1])
            elif pred == 'link':
                # (link ?l1 - location ?l2 - location)
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif pred == 'tightened' or pred == 'loose':
                # (tightened ?n - nut), (loose ?n - nut)
                self.nuts.add(parts[1])
            # Add other predicates if they existed and helped identify types

        # Build location graph from static facts
        self.location_graph = {loc: set() for loc in self.locations}
        for fact_str in task.static:
            parts = parse_fact(fact_str)
            if parts is None or len(parts) == 0: continue
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                # Ensure locations are known before adding links
                if l1 in self.location_graph and l2 in self.location_graph:
                     self.location_graph[l1].add(l2)
                     self.location_graph[l2].add(l1) # Assuming links are bidirectional

        # Compute shortest paths between all pairs of locations
        self.shortest_paths = compute_all_pairs_shortest_paths(self.location_graph)

        # Store static locations of nuts from initial state
        # Nut locations are assumed static based on domain structure
        for fact_str in task.initial_state:
            parts = parse_fact(fact_str)
            if parts is None or len(parts) == 0: continue
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.nuts:
                    self.nut_locations[obj] = loc

        # Store goal nuts
        for fact_str in task.goals:
             parts = parse_fact(fact_str)
             if parts is None or len(parts) == 0: continue
             if parts[0] == 'tightened':
                 self.goal_nuts.add(parts[1])


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state_facts = set(node.state)

        # --- Step 1: Extract relevant information from the current state ---
        man_loc = None
        U_loose = set() # Loose goal nuts
        S_carried = set() # Spanners carried by man
        S_usable = set() # Usable spanners (carried or on ground)
        spanner_current_locations = {} # Spanner -> current location (if on ground)

        # Assuming there is only one man object
        the_man = next(iter(self.men), None) # Get the single man object name

        for fact_str in state_facts:
            parts = parse_fact(fact_str)
            if parts is None or len(parts) == 0: continue

            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                if obj == the_man: # Check if the object is the man
                    man_loc = loc
                elif obj in self.spanners:
                    spanner_current_locations[obj] = loc
            elif pred == 'carrying':
                # Check if the fact is about the man carrying a spanner
                if parts[1] == the_man and parts[2] in self.spanners:
                    S_carried.add(parts[2])
            elif pred == 'usable':
                # spanner = parts[1]
                if parts[1] in self.spanners:
                    S_usable.add(parts[1])
            elif pred == 'loose':
                # nut = parts[1]
                if parts[1] in self.nuts and parts[1] in self.goal_nuts:
                    U_loose.add(parts[1])

        # Identify usable spanners carried and on ground
        S_usable_carried = S_carried & S_usable
        S_ground = {s for s in self.spanners if s not in S_carried}
        S_usable_ground = S_ground & S_usable

        # --- Step 2: Calculate heuristic components ---

        k_loose = len(U_loose)
        k_carried = len(S_usable_carried)
        k_ground = len(S_usable_ground)

        # Base case: Goal reached
        if k_loose == 0:
            return 0

        # Unsolvable case: Not enough usable spanners in the world
        if k_loose > k_carried + k_ground:
            return math.inf

        # Number of spanners that need to be picked up from the ground
        needed_pickups = max(0, k_loose - k_carried)

        # Identify locations relevant for required actions
        Nut_locs = {self.nut_locations[n] for n in U_loose if n in self.nut_locations} # Ensure nut location is known
        Spanner_locs = {spanner_current_locations[s] for s in S_usable_ground if s in spanner_current_locations} # Ensure spanner location is known

        Required_locs = set()
        Required_locs.update(Nut_locs)

        # Add locations of the 'needed_pickups' closest usable ground spanners
        if needed_pickups > 0 and Spanner_locs:
            # Need man_loc to be a valid key in shortest_paths
            if man_loc is None or man_loc not in self.shortest_paths:
                 # Man's location is unknown or not in the graph
                 return math.inf

            # Sort spanner locations by distance from man_loc
            # Use list() to make a mutable copy for sorting
            sorted_spanner_locs = sorted(list(Spanner_locs),
                                         key=lambda l: self.shortest_paths[man_loc].get(l, math.inf))

            # Add the locations of the needed_pickups closest spanners
            # Ensure we don't try to add more locations than available
            num_to_add = min(needed_pickups, len(sorted_spanner_locs))
            Required_locs.update(sorted_spanner_locs[:num_to_add])

            # If we needed pickups but couldn't find enough reachable spanner locations
            # This check is partly redundant with k_loose > k_carried + k_ground,
            # but handles cases where spanners exist but are unreachable.
            if needed_pickups > 0 and len(sorted_spanner_locs) < needed_pickups:
                 # This implies some needed spanners are unreachable
                 return math.inf


        # Calculate minimum travel cost to reach any required location
        travel_cost = 0
        if Required_locs:
            if man_loc is None or man_loc not in self.shortest_paths:
                 # Man's location is unknown or not in the graph
                 return math.inf

            min_dist = math.inf
            for loc in Required_locs:
                # Check if the required location is in the shortest_paths from man_loc
                if loc in self.shortest_paths[man_loc]:
                     min_dist = min(min_dist, self.shortest_paths[man_loc][loc])
                # else: loc is unreachable from man_loc, ignore this loc for min_dist,
                # but if ALL required locs are unreachable, min_dist remains inf.

            travel_cost = min_dist

            # If all required locations are unreachable
            if travel_cost == math.inf:
                 return math.inf
        # else: Required_locs is empty, travel_cost remains 0 (handled by k_loose == 0)


        # --- Step 3: Combine components for final heuristic value ---
        heuristic_value = k_loose + needed_pickups + travel_cost

        return heuristic_value
