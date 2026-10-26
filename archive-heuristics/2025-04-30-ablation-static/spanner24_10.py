import collections
import math

# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class spannerHeuristic: # Inherit from Heuristic if base class is provided
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the number of nuts to tighten, the number of spanners that need
    to be picked up, and the walking distance to the first necessary location
    (either a nut location if a spanner is carried, or a spanner location if one is needed).

    # Assumptions
    - All nuts that need tightening are initially loose and stay at their location.
    - Spanners become unusable after one use for tightening a nut.
    - The problem is solvable (enough usable spanners exist and locations are connected).
    - The cost of each action (walk, pickup, tighten) is 1.
    - There is exactly one man object, and its name can be identified (assumed 'bob' based on examples).

    # Heuristic Initialization
    - Extracts static facts, specifically `link` predicates, to build a graph
      of locations.
    - Computes all-pairs shortest paths between locations using BFS. This distance
      information is stored for efficient lookup during heuristic calculation.
    - Identifies the name of the man object (assumed 'bob').

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location (`man_location`).
    2. Identify all nuts that are currently loose and their locations (`loose_nuts_at_loc`).
    3. Identify all usable spanners, distinguishing between those carried by the man
       and those on the ground, noting their locations if on the ground (`carried_usable_spanners`, `ground_usable_spanners_at_loc`).
    4. Count the total number of loose nuts (`num_loose_nuts`). If 0, the heuristic is 0.
    5. Count the number of usable spanners currently carried by the man (`num_carried_usable`).
    6. Calculate the number of spanners that still need to be picked up from the ground
       to tighten all loose nuts: `needed_from_ground = max(0, num_loose_nuts - num_carried_usable)`.
    7. The base heuristic value is the sum of the minimum required `tighten_nut` actions
       and `pickup_spanner` actions: `num_loose_nuts + needed_from_ground`.
    8. Add the estimated walking cost. The man needs to walk to a location where he can
       make progress.
       - If the man is carrying at least one usable spanner (`num_carried_usable > 0`),
         the next useful location is where a loose nut is. Find the minimum distance
         from the man's current location to any location with a loose nut (`min_dist_to_nut`).
       - If the man is not carrying any usable spanner (`num_carried_usable == 0`)
         and there are loose nuts to tighten, he first needs a spanner. The next useful
         location is where a usable spanner is on the ground. Find the minimum distance
         from the man's current location to any location with a usable spanner on the ground (`min_dist_to_spanner`).
       - Add this minimum distance (`min_dist_to_nut` or `min_dist_to_spanner`) to the
         heuristic value. If no such location is reachable, the heuristic is infinity
         (indicating an unsolvable state).
    9. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and computing
        shortest path distances between locations.
        """
        # super().__init__(task) # Uncomment if inheriting from Heuristic base class
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Used to infer man's name if needed

        # Identify the man object name (assuming there's only one man)
        # This is a fragile assumption based on examples. A real parser would provide this.
        # We assume the man object is named 'bob' based on the provided examples.
        self.man_name = 'bob'


        # Build the location graph from static link facts
        self.location_graph = collections.defaultdict(set)
        self.all_locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "link" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Compute all-pairs shortest paths using BFS from each location
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all reachable nodes.
        Returns a dictionary mapping location to distance.
        """
        distances = {node: float('inf') for node in self.all_locations}
        if start_node not in self.all_locations:
             # Start node might not be in the graph if it's an isolated location
             # or if the graph is empty.
             return distances # Return all infinities

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.location_graph: # Check if current_loc has neighbors
                for neighbor in self.location_graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Lookup shortest path distance between two locations."""
        if loc1 in self.distances and loc2 in self.distances[loc1]:
            return self.distances[loc1][loc2]
        return float('inf') # Locations might be unreachable or not in the graph

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3 and parts[1] == self.man_name:
                 man_location = parts[2]
                 break

        if man_location is None:
             # Man must be at a location in a valid state.
             # If not found, this state is likely invalid or unreachable.
             return float('inf')


        # 2. Identify loose nuts and their locations
        loose_nuts_at_loc = collections.defaultdict(list)
        # We need to find all nuts that are loose AND find their location.
        # Iterate through state to find loose nuts first.
        loose_nuts_names = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "loose" and len(parts) == 2:
                loose_nuts_names.add(parts[1])

        # Now find locations for the loose nuts
        for nut in loose_nuts_names:
             nut_location = None
             for loc_fact in state:
                 loc_parts = get_parts(loc_fact)
                 if loc_parts[0] == "at" and len(loc_parts) == 3 and loc_parts[1] == nut:
                     nut_location = loc_parts[2]
                     break
             # If a loose nut's location is not found, it's an invalid state for this domain.
             # Assuming valid states where loose nuts are always at a location.
             if nut_location:
                 loose_nuts_at_loc[nut_location].append(nut)


        num_loose_nuts = sum(len(nuts) for nuts in loose_nuts_at_loc.values())

        # If no loose nuts, goal is reached
        if num_loose_nuts == 0:
            return 0

        # 3. Identify usable spanners (carried and on ground)
        carried_usable_spanners = set()
        ground_usable_spanners_at_loc = collections.defaultdict(set)
        usable_spanners = set()

        # First find all usable spanners
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "usable" and len(parts) == 2:
                spanner = parts[1]
                usable_spanners.add(spanner)

        # Then check if usable spanners are carried or on ground
        for spanner in usable_spanners:
             is_carried = False
             for fact in state:
                 parts = get_parts(fact)
                 if parts[0] == "carrying" and len(parts) == 3 and parts[1] == self.man_name and parts[2] == spanner:
                     carried_usable_spanners.add(spanner)
                     is_carried = True
                     break
             if not is_carried:
                 # Find location if on ground
                 spanner_location = None
                 for fact in state:
                     parts = get_parts(fact)
                     if parts[0] == "at" and len(parts) == 3 and parts[1] == spanner:
                         spanner_location = parts[2]
                         break
                 # If spanner_location is None, the usable spanner is neither carried nor at a location.
                 # Assuming valid states where usable spanners are either carried or at a location.
                 if spanner_location:
                     ground_usable_spanners_at_loc[spanner_location].add(spanner)


        num_carried_usable = len(carried_usable_spanners)
        num_ground_usable = sum(len(spanners) for spanners in ground_usable_spanners_at_loc.values())

        # 6. Calculate needed pickups
        needed_from_ground = max(0, num_loose_nuts - num_carried_usable)

        # Check solvability based on spanners
        if num_loose_nuts > num_carried_usable + num_ground_usable:
             # Not enough usable spanners in the world to tighten all nuts
             return float('inf') # Problem is unsolvable from this state

        # 7. Base heuristic: tighten + pickup actions
        h = num_loose_nuts + needed_from_ground

        # 8. Add walking cost
        min_dist_to_nut = float('inf')
        loose_nut_locations = set(loose_nuts_at_loc.keys()) # Get unique locations
        for loc in loose_nut_locations:
            dist = self.get_distance(man_location, loc)
            min_dist_to_nut = min(min_dist_to_nut, dist)

        min_dist_to_spanner = float('inf')
        ground_usable_spanner_locations = set(ground_usable_spanners_at_loc.keys()) # Get unique locations
        for loc in ground_usable_spanner_locations:
             dist = self.get_distance(man_location, loc)
             min_dist_to_spanner = min(min_dist_to_spanner, dist)

        walking_cost = float('inf')
        if num_carried_usable > 0:
            # Man has spanner, next target is a nut location
            # Only add walking cost if there are nuts to tighten and they are reachable
            if num_loose_nuts > 0 and min_dist_to_nut != float('inf'):
                 walking_cost = min_dist_to_nut
            # If num_loose_nuts > 0 but min_dist_to_nut is inf, walking_cost remains inf (unsolvable).
            # If num_loose_nuts == 0, we returned 0 already.

        else: # num_carried_usable == 0
            # Man needs spanner, next target is a usable spanner location on ground
            # Only add walking cost if there are spanners to pick up and they are reachable
            if needed_from_ground > 0 and min_dist_to_spanner != float('inf'):
                 walking_cost = min_dist_to_spanner
            # If needed_from_ground > 0 but min_dist_to_spanner is inf, walking_cost remains inf (unsolvable).
            # If needed_from_ground == 0, it implies num_loose_nuts <= num_carried_usable (which is 0), so num_loose_nuts must be 0.
            # This case (needed_from_ground == 0 and num_carried_usable == 0) only happens if num_loose_nuts == 0, which is handled.


        # If walking_cost is still infinity, it means required locations are unreachable
        if walking_cost == float('inf'):
             return float('inf')

        h += walking_cost

        return h
