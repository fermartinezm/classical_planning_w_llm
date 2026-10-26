# Assuming the environment provides the Heuristic base class
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, you might need a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass

from fnmatch import fnmatch
from collections import deque

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

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It sums the number of loose goal nuts (for tighten actions), the number of
    spanner pickups required, and an estimated travel cost for the man to visit
    all necessary locations (nut locations and spanner pickup locations).

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - A spanner is consumed (becomes unusable) after tightening one nut.
    - The man can only carry one spanner at a time.
    - Picking up a spanner requires the man to not be carrying one. (Implicit assumption based on typical domains and action structure).
    - Travel cost between linked locations is 1.
    - The man object is consistently named 'bob'.
    - Spanner objects are consistently named starting with 'spanner'.
    - Nut objects are consistently named starting with 'nut'.
    - Nut locations are static (do not change during planning).

    # Heuristic Initialization
    - Extracts goal nuts from the task goals.
    - Builds a graph of locations based on `link` facts from static information.
    - Computes all-pairs shortest paths between locations using BFS.
    - Initializes a cache for nut locations, which are found dynamically.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Loose Goal Nuts:** Find all nuts that are required to be `tightened` in the goal and are currently `loose` in the given state. Count them (`num_nuts_to_tighten`).
    2. **Check for Goal State:** If `num_nuts_to_tighten` is 0, the state is a goal state, and the heuristic value is 0.
    3. **Base Action Cost:** Initialize the heuristic value with `num_nuts_to_tighten` (representing the minimum number of `tighten_nut` actions required).
    4. **Locate Man and Spanners:** Determine the man's current location. Identify which spanner, if any, the man is carrying and whether it is `usable`. Find all `usable` spanners currently on the ground and their locations.
    5. **Check Spanner Availability:** Count the total number of `usable` spanners available (carried or on the ground). If this count is less than `num_nuts_to_tighten`, the problem is likely unsolvable with the available usable spanners; return a large heuristic value.
    6. **Calculate Pickup Actions:** Determine how many spanner `pickup_spanner` actions are needed. This is `max(0, num_nuts_to_tighten - (1 if man is currently carrying a usable spanner else 0))`. Add this count to the heuristic value.
    7. **Identify Required Locations:** The man must visit the location of each loose goal nut. Additionally, he must visit the locations of the `num_pickups_needed` closest usable spanners on the ground (closest to the man's current location) to pick them up. Collect these nut and spanner locations into a set of `locations_to_visit`.
    8. **Estimate Travel Cost:** Calculate an estimated travel cost for the man to visit all locations in `locations_to_visit`, starting from his current location. A greedy approach is used: repeatedly move to the closest unvisited location in the set until all are visited, summing the distances. Handle cases where required locations are unreachable (e.g., due to a disconnected graph) by returning a large value.
    9. **Total Heuristic:** Add the estimated travel cost to the heuristic value.
    10. **Return:** Return the calculated total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts, building the location
        graph, and computing shortest paths.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Store goal nuts (objects that need to be tightened)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "tightened"}

        # Build location graph and compute distances
        self.locations = set()
        self.adj = {} # Adjacency list for linked locations

        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.adj.setdefault(l1, []).append(l2)
                self.adj.setdefault(l2, []).append(l1) # Links are bidirectional

        self.dist = {} # Shortest path distances {start_loc: {end_loc: distance}}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

        # Cache for nut locations (assuming nuts are static)
        self._nut_locations = {}


    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            curr_loc = queue.popleft()
            if curr_loc in self.adj:
                for neighbor in self.adj[curr_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[curr_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find man's location and name (assuming man is 'bob')
        man_loc = None
        man_name = 'bob' # Assuming man is named 'bob' based on example
        for fact in state:
            if match(fact, "at", man_name, "*"):
                 man_loc = get_parts(fact)[2]
                 break

        # If man's location isn't found, something is wrong with the state representation
        # or the assumption about the man's name. Return a large value.
        if man_loc is None:
             return 1000000 # Indicate invalid state

        # Find loose goal nuts in the current state
        loose_goal_nuts_in_state = {
            nut for nut in self.goal_nuts
            if f"(loose {nut})" in state
        }

        num_nuts_to_tighten = len(loose_goal_nuts_in_state)

        # If all goal nuts are tightened, heuristic is 0
        if num_nuts_to_tighten == 0:
            return 0

        # Base cost: tighten actions
        h = num_nuts_to_tighten

        # Find usable spanners
        usable_carried_spanner = None
        usable_spanners_on_ground = {} # {spanner_name: location}

        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "carrying" and parts[1] == man_name:
                 spanner = parts[2]
                 if f"(usable {spanner})" in state:
                     usable_carried_spanner = spanner
             elif parts[0] == "at" and parts[1].startswith('spanner'): # Assuming spanners start with 'spanner'
                 spanner, loc = parts[1:]
                 if f"(usable {spanner})" in state:
                     # Check if this usable spanner is the one being carried
                     is_carried = False
                     for f_carried in state:
                         if match(f_carried, "carrying", man_name, spanner):
                             is_carried = True
                             break
                     if not is_carried:
                         usable_spanners_on_ground[spanner] = loc


        num_usable_available = len(usable_spanners_on_ground) + (1 if usable_carried_spanner is not None else 0)

        # Check if enough usable spanners exist
        if num_nuts_to_tighten > num_usable_available:
             # Problem is likely unsolvable with available usable spanners
             return 1000000 # Large value

        # Calculate number of pickups needed
        num_pickups_needed = max(0, num_nuts_to_tighten - (1 if usable_carried_spanner is not None else 0))
        h += num_pickups_needed # Cost for pickup actions

        # Estimate travel cost
        # Man needs to visit all nut locations and num_pickups_needed spanner locations

        # Find locations of loose goal nuts
        nut_locations = set()
        for nut in loose_goal_nuts_in_state:
             # Find the location of this nut in the current state (assuming static location)
             nut_loc = self._nut_locations.get(nut)
             if nut_loc is None:
                 # Find and cache the nut location from the current state
                 for fact in state:
                     if match(fact, "at", nut, "*"):
                         nut_loc = get_parts(fact)[2]
                         self._nut_locations[nut] = nut_loc # Store it
                         break
             if nut_loc:
                 nut_locations.add(nut_loc)
             # else: nut location not found? Problem state? This shouldn't happen if nuts are locatable.
             # If a nut location is not found, it implies an invalid state or domain definition.
             # We could return a large value here, but assuming valid states.


        # Find locations of usable spanners on the ground, sorted by distance from man
        # Ensure man_loc is in dist map before accessing
        if man_loc not in self.dist:
             # Man is in a location not part of the linked graph? Unreachable?
             return 1000000 # Indicate invalid state

        available_spanner_locs_list = sorted(
            usable_spanners_on_ground.values(),
            key=lambda l: self.dist[man_loc].get(l, float('inf')) # Use .get for safety
        )

        # Select the locations of the closest spanners for pickup
        spanner_pickup_locations_needed = available_spanner_locs_list[:num_pickups_needed]

        # Set of locations the man must visit
        locations_to_visit = set(nut_locations)
        locations_to_visit.update(spanner_pickup_locations_needed)

        # Calculate greedy travel cost
        current_loc = man_loc
        travel_cost = 0
        locations_to_visit_list = list(locations_to_visit) # Convert to list for easier modification

        while locations_to_visit_list:
            # Find the closest location to the current location among those to visit
            next_loc = None
            min_dist = float('inf')

            for loc in locations_to_visit_list:
                # Ensure the location is reachable from the current location
                # Check if current_loc is in self.dist and loc is in self.dist[current_loc]
                if current_loc not in self.dist or loc not in self.dist[current_loc]:
                     # A required location is unreachable from the current location
                     return 1000000 # Indicate unsolvability

                distance = self.dist[current_loc][loc]
                if distance < min_dist:
                    min_dist = distance
                    next_loc = loc

            if next_loc is None:
                 # Should not happen if locations_to_visit_list is not empty
                 # and all locations are reachable.
                 break # Safety break

            travel_cost += min_dist
            current_loc = next_loc
            locations_to_visit_list.remove(next_loc)

        h += travel_cost

        return h
