# Assuming Heuristic base class is available and has __init__(self, task) and __call__(self, node)
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not isinstance(fact, str) or not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.
    Returns a dictionary {node: distance}. Unreachable nodes have distance infinity.
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
         # Start node must be in the graph to start BFS
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current = queue.popleft()

        # If current node is unreachable (shouldn't happen if added to queue with finite distance), skip
        if distances[current] == math.inf:
            continue

        for neighbor in graph.get(current, []):
            if distances[neighbor] == math.inf: # Check if visited
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances

class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose goal nuts. It sums the estimated costs for tighten actions, pickup
    actions, and movement actions. Movement cost is estimated as the sum of
    shortest path distances from the man's current location to all required
    locations (loose nut locations and locations of needed spanners).

    # Assumptions:
    - The man can carry at most one spanner at a time. Picking up a spanner
      while carrying one implicitly drops the old one at the pickup location.
      (Note: The PDDL is ambiguous here, but this is a common interpretation).
    - Each usable spanner can tighten exactly one nut.
    - The graph of locations defined by `link` predicates is directed as specified.
    - All goal nuts are initially loose and remain at their initial locations.
    - All usable spanners remain at their initial locations until picked up.
    - All locations and objects are correctly typed and referenced in the PDDL.
    - The man object can be identified (e.g., by being involved in a 'carrying'
      fact in the initial state, or defaulting to 'bob').

    # Heuristic Initialization
    - Identify all goal nuts from the task goals.
    - Build the graph of locations based on `link` predicates found in static facts.
    - Identify all locations mentioned in initial state `at` facts and static `link` facts.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Store initial locations of all objects (nuts, spanners, man) for quick lookup.
    - Identify the name of the man object.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location (`L_man`).
    2. Identify all loose nuts in the current state that are also goal nuts (`LooseGoalNuts`).
    3. For each `N` in `LooseGoalNuts`, find its location (`L_nut`). Store these in `LooseNutLocs`.
    4. Identify all usable spanners in the current state (`UsableSpanners`).
    5. Separate usable spanners into those the man is carrying (`CarriedUsableSpanners`) and those on the ground (`UsableSpannersOnGround`).
    6. For each `S` in `UsableSpannersOnGround`, find its location (`L_s`). Store these in `UsableSpannerLocs`.
    7. Count the number of loose goal nuts (`num_loose_goal_nuts`).
    8. Count the number of usable spanners the man is carrying (`num_carried_usable_spanners`).
    9. Count the number of usable spanners on the ground (`num_usable_spanners_on_ground`).
    10. Check for unsolvability: If `num_loose_goal_nuts > num_carried_usable_spanners + num_usable_spanners_on_ground`, the goal is unreachable. Return `math.inf`.
    11. Calculate the number of additional spanners the man needs to pick up from the ground:
        `num_spanners_to_pickup = max(0, num_loose_goal_nuts - num_carried_usable_spanners)`.
    12. The heuristic is the sum of:
        - The number of `tighten_nut` actions needed (`num_loose_goal_nuts`).
        - The number of `pickup_spanner` actions needed (`num_spanners_to_pickup`).
        - An estimate of the `walk` actions needed.
    13. Estimate movement cost: The man needs to visit the location of each loose goal nut and the location of each spanner he needs to pick up.
        - The set of required nut locations is `LooseNutLocs`.
        - The set of required spanner locations is the locations of the `num_spanners_to_pickup` usable spanners on the ground that are closest to the man's current location.
        - The total set of required locations to visit is `RequiredLocs = LooseNutLocs` union the selected spanner locations.
        - Movement cost is estimated as the sum of shortest distances from the man's current location (`L_man`) to each location in `RequiredLocs`. If `L_man` cannot reach any required location, the goal is unreachable.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "tightened":
                self.goal_nuts.add(parts[1])

        # 2. Build location graph and identify all locations
        self.location_graph = {}
        all_locations = set()

        # Add locations from links
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                self.location_graph[loc1].append(loc2)

        # Add locations from initial 'at' facts if not already included
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at":
                 location = parts[2]
                 all_locations.add(location)
                 if location not in self.location_graph:
                     self.location_graph[location] = [] # Add isolated locations to graph nodes

        # Ensure all locations are nodes in the graph dictionary
        for loc in all_locations:
             if loc not in self.location_graph:
                 self.location_graph[loc] = []

        # 3. Precompute all-pairs shortest paths
        self.all_pairs_distances = {}
        for start_loc in self.location_graph:
            self.all_pairs_distances[start_loc] = bfs(self.location_graph, start_loc)

        # Store initial object locations (useful for static objects like nuts)
        self.initial_object_locations = {}
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "at":
                obj, loc = parts[1], parts[2]
                self.initial_object_locations[obj] = loc

        # Identify the man object name (assuming there's only one man)
        self.man_name = None
        # Try finding the object involved in a 'carrying' fact in the initial state
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "carrying":
                 self.man_name = parts[1]
                 break

        # Fallback: Assume man is named 'bob' if not found via 'carrying' in initial state
        if self.man_name is None:
             # This is a heuristic-specific assumption based on examples
             self.man_name = "bob"


    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if unreachable or location unknown."""
        if loc1 not in self.all_pairs_distances or loc2 not in self.all_pairs_distances.get(loc1, {}):
             # This can happen if loc1 or loc2 is not a recognized location node
             return math.inf
        return self.all_pairs_distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None:
             # Man is not at any location? Problem state.
             return math.inf

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts = set()
        loose_nut_locs = set()
        current_tightened_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "tightened" and parts[1] in self.goal_nuts:
                current_tightened_nuts.add(parts[1])

        for nut in self.goal_nuts:
            if nut not in current_tightened_nuts:
                 # It's a goal nut and not yet tightened.
                 loose_goal_nuts.add(nut)
                 # Get nut location - should be static
                 if nut in self.initial_object_locations:
                     loose_nut_locs.add(self.initial_object_locations[nut])
                 else:
                     # Cannot find initial location of a goal nut, problem likely malformed
                     return math.inf


        # 7. Count loose goal nuts
        num_loose_goal_nuts = len(loose_goal_nuts)

        # If no loose goal nuts, we are in a goal state
        if num_loose_goal_nuts == 0:
            return 0

        # 3. Identify usable spanners (carried or on ground)
        usable_spanners = set()
        carried_usable_spanners = set()
        usable_spanners_on_ground = {} # {spanner: location}
        current_spanner_locations = {} # To find spanner location in current state

        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "usable":
                    usable_spanners.add(parts[1])
                elif parts[0] == "at" and parts[1] in usable_spanners: # Check if usable spanner is at a location
                     current_spanner_locations[parts[1]] = parts[2]
                elif parts[0] == "carrying" and parts[2] == self.man_name and parts[1] in usable_spanners: # Check if usable spanner is carried by man
                     carried_usable_spanners.add(parts[1])


        for spanner in usable_spanners:
            if spanner not in carried_usable_spanners:
                 # Usable spanner is not carried, check if it's on the ground
                 if spanner in current_spanner_locations:
                      usable_spanners_on_ground[spanner] = current_spanner_locations[spanner]
                 else:
                      # Usable spanner exists but is neither carried nor on ground? Problem state.
                      return math.inf


        # 8. Count carried usable spanners
        num_carried_usable_spanners = len(carried_usable_spanners)

        # 9. Count usable spanners on ground
        num_usable_spanners_on_ground = len(usable_spanners_on_ground)

        # 10. Check for unsolvability (total spanners available)
        if num_loose_goal_nuts > num_carried_usable_spanners + num_usable_spanners_on_ground:
            return math.inf # Not enough usable spanners in the world

        # 11. Calculate spanners to pickup
        num_spanners_to_pickup = max(0, num_loose_goal_nuts - num_carried_usable_spanners)

        # 12. Base cost (tighten + pickup actions)
        base_cost = num_loose_goal_nuts + num_spanners_to_pickup

        # 13. Estimate movement cost
        required_locations = set(loose_nut_locs)

        # Find the locations of the closest usable spanners on the ground
        spanner_locs_on_ground = list(usable_spanners_on_ground.values())

        # Calculate distances from man's current location to all spanner locations on ground
        distances_to_spanners = []
        for s_loc in spanner_locs_on_ground:
             dist = self.get_distance(man_location, s_loc)
             # We only consider reachable spanners for pickup calculation
             if dist != math.inf:
                 distances_to_spanners.append((dist, s_loc))

        # Sort reachable spanner locations by distance and take the closest ones needed
        distances_to_spanners.sort()
        closest_spanner_locs_needed = [loc for dist, loc in distances_to_spanners[:num_spanners_to_pickup]]

        # Check if enough reachable spanners exist for pickup
        if len(closest_spanner_locs_needed) < num_spanners_to_pickup:
             # We need to pick up N spanners, but fewer than N reachable usable spanners exist on the ground
             return math.inf


        required_locations.update(closest_spanner_locs_needed)

        # Calculate sum of distances from man's location to each required location
        movement_cost = 0
        for loc in required_locations:
            dist = self.get_distance(man_location, loc)
            if dist == math.inf:
                # A required location (nut or needed spanner) is unreachable
                return math.inf
            movement_cost += dist

        # Total heuristic value
        total_cost = base_cost + movement_cost

        return total_cost
