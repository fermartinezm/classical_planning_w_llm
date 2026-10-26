# Ensure necessary imports
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS from a start node to find shortest distances to all reachable nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Start node is not in the graph (e.g., isolated location not linked)
        # Cannot reach anything from here.
        return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the number of nuts still needing tightening, the cost to acquire a
    usable spanner if not already held, and the travel cost to reach the locations
    of the loose nuts.

    # Assumptions
    - The agent (bob) needs a usable spanner to tighten a nut.
    - The agent must be at the same location as the nut to tighten it.
    - The location graph defined by 'link' predicates is undirected.
    - All necessary objects (bob, spanners, nuts, locations) exist and are relevant.
    - The problem is solvable (i.e., usable spanners exist, locations are connected).

    # Heuristic Initialization
    - Builds the location graph based on 'link' facts from the static information.
    - Pre-calculates shortest path distances between all pairs of locations using BFS.
    - Identifies the set of nuts that need to be tightened from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify Bob's current location.
    2. Determine if Bob is currently carrying a usable spanner.
    3. Identify all nuts that are currently loose and are part of the goal, and find their locations.
    4. If no goal nuts are loose, the heuristic is 0 (goal reached for nuts).
    5. Calculate the cost to acquire a usable spanner if Bob is not carrying one:
       - Find usable spanners currently on the ground and their locations.
       - Find the closest usable spanner to Bob's current location using pre-calculated distances.
       - The cost is the distance to the closest spanner location plus 1 (for the pickup action). If Bob is already carrying a usable spanner, this cost is 0.
       - Determine the 'effective start location' for subsequent travel: This is Bob's current location if he has a spanner, or the location of the closest spanner after picking it up. If no usable spanners are available or reachable, the problem is likely unsolvable, and the heuristic should reflect this (e.g., return infinity).
    6. Calculate the travel cost to reach the loose nuts:
       - Find the maximum shortest distance from the 'effective start location' to the location of any loose goal nut. This estimates the longest single trip needed to get "into the vicinity" of the work. If any loose nut location is unreachable, the problem is likely unsolvable, and the heuristic should reflect this.
    7. The total heuristic value is the sum of:
       - The number of loose goal nuts (representing the 'tighten' action for each).
       - The cost to acquire a spanner (calculated in step 5).
       - The travel cost to reach the loose nuts (calculated in step 6).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and building
        the location graph and distance matrix.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state to find all locations

        # Identify all locations from static links and initial state 'at' facts
        all_locations = set()
        self.graph = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Assuming links are bidirectional

        # Add locations from initial state 'at' facts that might not be linked
        # This ensures all locations Bob, spanners, or nuts might be at are included
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 obj, loc = parts[1], parts[2]
                 all_locations.add(loc)
                 self.graph.setdefault(loc, []) # Ensure location exists in graph even if no links

        # Ensure all locations found are keys in the graph dictionary
        for loc in all_locations:
             self.graph.setdefault(loc, [])

        # Pre-calculate all-pairs shortest paths
        self.distances = {}
        for start_loc in self.graph:
            self.distances[start_loc] = bfs(self.graph, start_loc)

        # Store the set of nuts that need to be tightened in the goal
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                self.goal_nuts.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Bob's current location.
        bob_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_location = get_parts(fact)[2]
                break
        if bob_location is None:
             # Bob must always be somewhere in a solvable problem
             return float('inf') # Should not happen in valid states

        # 2. Determine if Bob is currently carrying a usable spanner.
        carried_spanners = set()
        usable_spanners_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "carrying" and parts[1] == "bob":
                carried_spanners.add(parts[2])
            elif parts[0] == "usable":
                 usable_spanners_in_state.add(parts[1])

        bob_has_usable_spanner = any(s in usable_spanners_in_state for s in carried_spanners)

        # 3. Identify all nuts that are currently loose and are part of the goal, and find their locations.
        loose_goal_nuts = {} # {nut: location}
        nut_locations = {} # {nut: location} - temporary mapping for all nuts
        loose_nuts_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in self.goal_nuts:
                 nut_locations[parts[1]] = parts[2]
            elif parts[0] == "loose" and parts[1] in self.goal_nuts:
                 loose_nuts_in_state.add(parts[1])

        # Now combine loose status with location for goal nuts
        for nut in self.goal_nuts:
             if nut in loose_nuts_in_state and nut in nut_locations:
                  loose_goal_nuts[nut] = nut_locations[nut]


        # 4. If no goal nuts are loose, the heuristic is 0.
        num_loose_nuts = len(loose_goal_nuts)
        if num_loose_nuts == 0:
            return 0

        # 5. Calculate the cost to acquire a usable spanner if Bob is not carrying one.
        spanner_cost = 0
        effective_start_location = bob_location

        if not bob_has_usable_spanner:
            # Find usable spanners currently on the ground
            usable_spanners_on_ground = {} # {spanner: location}
            for fact in state:
                 parts = get_parts(fact)
                 # Check if the object is a usable spanner and is at a location
                 if parts[0] == "at" and parts[1] in usable_spanners_in_state:
                      usable_spanners_on_ground[parts[1]] = parts[2]

            min_dist_to_spanner = float('inf')
            closest_spanner_location = None

            for spanner, loc in usable_spanners_on_ground.items():
                 # Ensure both bob_location and spanner location are in the pre-calculated distances
                 if bob_location in self.distances and loc in self.distances[bob_location]:
                    dist = self.distances[bob_location][loc]
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        closest_spanner_location = loc

            if closest_spanner_location is None or min_dist_to_spanner == float('inf'):
                 # No usable spanners found on the ground or reachable from Bob's location
                 # Problem likely unsolvable.
                 return float('inf')

            spanner_cost = min_dist_to_spanner + 1 # Move to spanner + pickup
            effective_start_location = closest_spanner_location


        # 6. Calculate the travel cost to reach the loose nuts.
        travel_cost = 0
        if num_loose_nuts > 0:
            max_dist_to_nut_location = 0
            for nut_loc in loose_goal_nuts.values():
                 # Ensure effective_start_location and nut_loc are in the pre-calculated distances
                 if effective_start_location in self.distances and nut_loc in self.distances[effective_start_location]:
                    dist = self.distances[effective_start_location][nut_loc]
                    max_dist_to_nut_location = max(max_dist_to_nut_location, dist)
                 else:
                     # A loose nut is at an unreachable location from the effective start location
                     # Problem unsolvable.
                     return float('inf')
            travel_cost = max_dist_to_nut_location

        # 7. Total heuristic value
        # Number of tighten actions + Spanner acquisition cost + Travel cost
        total_cost = num_loose_nuts + spanner_cost + travel_cost

        return total_cost
