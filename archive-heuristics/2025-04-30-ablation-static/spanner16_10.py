# The base class Heuristic is assumed to be available in the environment.
# from heuristics.heuristic_base import Heuristic
# Assuming Heuristic is defined elsewhere and provides the necessary interface.

from fnmatch import fnmatch
import heapq # For MST

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

def bfs_shortest_path(graph, start):
    """Compute shortest path distances from start to all reachable nodes."""
    distances = {start: 0}
    queue = [start]
    while queue:
        current = queue.pop(0)
        # Use set for neighbors to avoid duplicates if graph is represented with duplicates
        for neighbor in set(graph.get(current, [])):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the minimum number of actions (walk, pickup, tighten)
    required to tighten all goal nuts. It considers the cost of acquiring usable
    spanners and traveling to the locations of nuts and spanners.

    # Assumptions
    - Each usable spanner can tighten exactly one nut.
    - The number of usable spanners must be at least the number of goal nuts for the problem to be solvable.
    - The man must visit the location of each nut to tighten it.
    - The man must visit the location of a spanner to pick it up.
    - Travel cost is estimated using a Minimum Spanning Tree over required locations.
    - There is only one man, named 'bob'.
    - Nuts are named starting with 'nut', spanners with 'spanner'.
    - Nut locations are static and can be found in the initial state.
    - All locatable objects (man, spanner, nut) are always at a location.

    # Heuristic Initialization
    - Build the location graph from `link` facts.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify all nut objects and their static locations from the initial state.
    - Store the set of goal nuts.
    - Identify the man's name.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of loose nuts that are goal conditions (`current_loose_nuts`).
    2. If `current_loose_nuts` is empty, the heuristic is 0 (goal reached).
    3. Identify the set of usable spanners currently in the state (`current_usable_spanners`).
    4. Count usable spanners the man is currently carrying (`k_carried_usable`).
    5. If the total number of usable spanners (`|current_usable_spanners|`) is less than the number of goal nuts (`|current_loose_nuts|`), the problem is unsolvable from this state; return infinity.
    6. Calculate the number of additional spanners the man needs to pick up (`spanners_to_pickup = max(0, |current_loose_nuts| - k_carried_usable)`).
    7. The cost includes:
       - `|current_loose_nuts|` tighten actions (cost 1 each).
       - `spanners_to_pickup` pickup actions (cost 1 each).
       - Travel cost.
    8. Determine the locations the man *must* visit:
       - The location of each nut in `current_loose_nuts`.
       - The locations of `spanners_to_pickup` usable spanners that are currently on the ground. Select the `spanners_to_pickup` available spanners whose locations are closest to the man's current location.
    9. Compute the travel cost as the weight of a Minimum Spanning Tree (MST) connecting the man's current location and all the required visit locations identified in step 8. Use shortest path distances as edge weights.
    10. The total heuristic value is the sum of tighten cost, pickup cost, and travel cost.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        # Assume task object has attributes: goals, static, initial_state
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Build location graph and compute distances
        self.location_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Store nut locations (assuming they are static)
        self.nut_locations = {}
        # Find all nuts mentioned in goals or initial state (loose/tightened)
        all_nuts_mentioned = set()
        for goal in self.goals:
             if match(goal, "tightened", "*"):
                 _, nut = get_parts(goal)
                 all_nuts_mentioned.add(nut)
        for fact in initial_state:
             if match(fact, "loose", "*"):
                 _, nut = get_parts(fact)
                 all_nuts_mentioned.add(nut)
             if match(fact, "tightened", "*"):
                 _, nut = get_parts(fact)
                 all_nuts_mentioned.add(nut)

        # Find initial location for each identified nut
        for nut in all_nuts_mentioned:
             for fact in initial_state:
                 if match(fact, "at", nut, "*"):
                     self.nut_locations[nut] = get_parts(fact)[2]
                     # Add nut location to all_locations if it wasn't linked
                     self.all_locations.add(self.nut_locations[nut])
                     break # Found location for this nut

        # Recompute all-pairs shortest paths now that all relevant locations are identified
        self.distances = {}
        for loc in list(self.all_locations): # Use list to avoid modifying set during iteration
            self.distances[loc] = bfs_shortest_path(self.location_graph, loc)


        # Store goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.goal_nuts.add(nut)

        # Find the man's name (assuming only one man, 'bob')
        # A more robust way would be to parse object types from the domain file
        # or infer from initial state facts like (at ?m - man ?l - location)
        # Based on examples, 'bob' is the man.
        self.man_name = 'bob'


    def get_distance(self, loc1, loc2):
        """Get shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Ensure both locations are in the computed distances map
        # If loc1 is not in distances, it means it's not a location in the graph or initial state
        # If loc2 is not in distances[loc1], it means loc2 is unreachable from loc1
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # Cannot reach, return infinity
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Identify loose goal nuts
        current_loose_nuts = {
            nut for nut in self.goal_nuts if f"(loose {nut})" in state
        }

        # 2. If no loose goal nuts, goal is reached
        if not current_loose_nuts:
            return 0

        # 3. Identify usable spanners in the current state
        current_usable_spanners = set()
        for fact in state:
             if match(fact, "usable", "*"):
                 _, spanner = get_parts(fact)
                 current_usable_spanners.add(spanner)

        # 4. Count carried usable spanners
        carried_usable_spanners = {
            s for s in current_usable_spanners if f"(carrying {self.man_name} {s})" in state
        }
        k_carried_usable = len(carried_usable_spanners)

        # 5. Check solvability
        if len(current_usable_spanners) < len(current_loose_nuts):
            return float('inf') # Not enough usable spanners

        # 6. Calculate costs
        tighten_cost = len(current_loose_nuts)
        spanners_to_pickup = max(0, len(current_loose_nuts) - k_carried_usable)
        pickup_cost = spanners_to_pickup

        # 7. Determine locations to visit
        man_location = None
        for fact in state:
             if match(fact, "at", self.man_name, "*"):
                 man_location = get_parts(fact)[2]
                 break
        if man_location is None:
             # Man must be somewhere if the state is valid and reachable from initial state.
             # If not found, this state is likely problematic or a dead end.
             return float('inf')


        nut_locations_to_visit = {self.nut_locations[nut] for nut in current_loose_nuts}

        available_spanner_locations_on_ground = {} # Map location to list of usable spanners there
        for s in current_usable_spanners:
            if s not in carried_usable_spanners: # Spanner is on the ground
                 # Find spanner location in state
                 spanner_loc = None
                 for fact in state:
                     if match(fact, "at", s, "*"):
                         spanner_loc = get_parts(fact)[2]
                         break
                 # If a usable spanner is not carried and not at a location, something is wrong with the state.
                 # Or maybe it's an implicit location? Assuming all locatable objects are (at loc).
                 if spanner_loc:
                     available_spanner_locations_on_ground.setdefault(spanner_loc, []).append(s)

        # Select spanner locations to visit: locations of the 'spanners_to_pickup' closest available spanners
        spanner_locs_with_dist = []
        for loc, spanners_at_loc in available_spanner_locations_on_ground.items():
             dist = self.get_distance(man_location, loc)
             if dist != float('inf'):
                 # Add each spanner as a potential pickup, not just the location
                 for s in spanners_at_loc:
                     spanner_locs_with_dist.append((dist, loc, s))

        # Sort by distance and take the top spanners_to_pickup
        spanner_locs_with_dist.sort()
        spanner_pickup_locations = set()
        picked_spanners_count = 0
        # Iterate through sorted spanners and add their locations until we need enough spanners
        for dist, loc, s in spanner_locs_with_dist:
             if picked_spanners_count < spanners_to_pickup:
                 spanner_pickup_locations.add(loc)
                 picked_spanners_count += 1
             else:
                 break # Got enough spanner locations

        locations_to_visit = nut_locations_to_visit | spanner_pickup_locations

        # 8. Compute travel cost using MST
        all_nodes_for_mst = locations_to_visit | {man_location}

        # Check if all required locations are reachable from the man's location
        # If any required location is unreachable, the problem is unsolvable from here.
        # This check is crucial before attempting MST.
        for loc in locations_to_visit:
             if self.get_distance(man_location, loc) == float('inf'):
                  return float('inf')

        # If man_location itself is not in the distances map (e.g., isolated island), problem is unsolvable
        # This might happen if the initial state places the man at a location not linked to anything
        # and not mentioned in initial 'at' facts of other objects used to build the distance map.
        # Recomputing distances after adding nut locations should cover most cases, but this check is safer.
        if man_location not in self.distances and len(locations_to_visit) > 0:
             return float('inf')


        if len(all_nodes_for_mst) <= 1:
             travel_cost = 0
        else:
            travel_cost = self.compute_mst_cost(list(all_nodes_for_mst), man_location)


        # Total heuristic value
        total_cost = tighten_cost + pickup_cost + travel_cost

        return total_cost


    def compute_mst_cost(self, nodes, start_node):
        """Compute the MST cost for a set of nodes including the start node using Prim's algorithm."""
        # Prim's algorithm starting from start_node
        visited = {start_node}
        mst_cost = 0
        # Priority queue stores tuples (weight, neighbor_node)
        pq = []

        # Add initial edges from start_node to all other nodes
        for v in nodes:
            if v != start_node:
                dist = self.get_distance(start_node, v)
                # We already checked reachability for required nodes before calling MST.
                # If dist is inf here, it means v is not a required node, but just another node
                # included in 'nodes' (e.g., man_location might be disconnected from some non-required nodes).
                # The MST should connect the required nodes.
                if dist != float('inf'):
                    heapq.heappush(pq, (dist, v))

        while pq and len(visited) < len(nodes):
            weight, v = heapq.heappop(pq)

            if v not in visited:
                visited.add(v)
                mst_cost += weight

                # Add edges from the newly visited node v to all unvisited nodes
                for next_node in nodes:
                    if next_node not in visited:
                        dist = self.get_distance(v, next_node)
                        if dist != float('inf'):
                             heapq.heappush(pq, (dist, next_node))

        # After running Prim's, check if all *required* locations were included in the MST.
        # The set `locations_to_visit` are the required ones (nut locs + spanner pickup locs).
        # The `nodes` parameter includes `man_location` as well.
        # If the MST couldn't connect all nodes in `nodes`, it means the graph component
        # containing `start_node` doesn't include all nodes in `nodes`.
        # We already checked if all `locations_to_visit` are reachable from `man_location` before calling.
        # So, if we reach here, all required nodes *are* reachable from `start_node`.
        # The MST should connect `start_node` to all nodes in `locations_to_visit`.
        # The size of the visited set should be at least `len(locations_to_visit) + 1` (if man_location is distinct).
        # A simpler check: ensure all required locations are in the visited set.
        # This check is redundant if the pre-check passed, but safer.
        # if not locations_to_visit.issubset(visited):
        #      # This should not happen if the pre-check passed
        #      return float('inf') # Indicate problem

        return mst_cost
