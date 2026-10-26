from collections import deque, defaultdict
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions (as seen in Logistics example)
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total number of actions required to tighten all loose nuts.
    It sums the number of tighten actions, the number of spanner pickup actions needed,
    and an estimate of the travel cost. The travel cost is estimated as the sum of
    distances from the man's current location to each loose nut location, plus the sum
    of distances from the man's current location to the locations of the closest usable
    spanners that need to be picked up.

    # Assumptions
    - Each loose nut must be tightened (1 tighten action).
    - Each tightening requires a usable spanner. A spanner becomes unusable after one use.
    - The man can carry multiple spanners (implied by predicate structure).
    - If the man needs N usable spanners in total for the remaining nuts and carries M usable ones, he needs to pick up max(0, N-M) more from the ground (1 pickup action each).
    - Shortest path distances between locations represent the minimum walk actions.
    - The set of locations and connectivity is defined by the static 'link' facts.

    # Heuristic Initialization
    - Precomputes shortest path distances between all pairs of locations based on `link` facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location (`ManLoc`). This requires identifying the man object first. The man is assumed to be the locatable object that is not a spanner or nut.
    2. Identify usable spanners: those carried by the man (`UsableCarried`) and those on the ground (`UsableOnGround`).
    3. Identify loose nuts and their locations (`LooseNuts`).
    4. If there are no loose nuts, the heuristic is 0 (goal state).
    5. Calculate the number of loose nuts (`NumLoose`). This is the minimum number of tighten actions required.
    6. Calculate the number of usable spanners currently carried (`NumUsableCarried`).
    7. Calculate the number of spanners that need to be picked up from the ground (`NumToPickup = max(0, NumLoose - NumUsableCarried)`). This is the minimum number of pickup actions from the ground required to have enough spanners for all loose nuts.
    8. Initialize `total_heuristic_cost = NumLoose + NumToPickup`. This accounts for the tighten and pickup actions.
    9. Estimate travel cost:
       a. Add the sum of shortest path distances from `ManLoc` to the location of each loose nut. This estimates the travel needed to reach each nut.
       b. Identify the `NumToPickup` usable spanners on the ground that are closest to `ManLoc`. Add the sum of shortest path distances from `ManLoc` to the locations of these spanners. This estimates the travel needed to reach the spanners that must be picked up.
    10. Return `total_heuristic_cost`.

    This heuristic is non-admissible as it sums distances independently, potentially overestimating
    travel by assuming separate trips for each nut and spanner pickup. However, it aims to
    provide a strong gradient towards the goal for greedy search by capturing the total
    number of required actions and the cost to reach the necessary objects.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing shortest path distances
        based on static link facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract locations and links from static facts
        self.locations = set()
        self.links = defaultdict(set)
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.links[l1].add(l2)
                self.links[l2].add(l1) # Links are bidirectional

        # Precompute all-pairs shortest paths using BFS
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS starting from each location defined by links.
        Returns a dictionary distances[start_loc][end_loc] = distance.
        """
        distances = {}
        for start_node in self.locations:
            distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_node, dist = queue.popleft()
                distances[start_node][current_node] = dist

                for neighbor in self.links.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return distances

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Return infinity if locations are not in the precomputed graph
        # or if there is no path.
        return self.distances.get(loc1, {}).get(loc2, float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location and name
        man_location = None
        man_name = None
        # Assuming man is the object that is 'at' a location and is not a spanner or nut
        # This is a simplification based on typical domain structures.
        # A more robust approach would involve parsing object types.
        all_locatables = set()
        spanners_nuts = set()

        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj = get_parts(fact)[1]
                  all_locatables.add(obj)
             if match(fact, "usable", "*"):
                  spanners_nuts.add(get_parts(fact)[1])
             if match(fact, "loose", "*"):
                  spanners_nuts.add(get_parts(fact)[1])
             if match(fact, "tightened", "*"):
                  spanners_nuts.add(get_parts(fact)[1])
             if match(fact, "carrying", "*", "*"):
                  man_candidate = get_parts(fact)[1]
                  carried_obj = get_parts(fact)[2]
                  all_locatables.add(man_candidate)
                  spanners_nuts.add(carried_obj)

        man_candidates = all_locatables - spanners_nuts
        if len(man_candidates) >= 1: # Assume at least one man exists
             man_name = list(man_candidates)[0] # Pick the first one if multiple
             # Find man's location
             for fact in state:
                  if match(fact, "at", man_name, "*"):
                       man_location = get_parts(fact)[2]
                       break
        # If man_name or man_location is still None, it's a problematic state
        if man_name is None or man_location is None:
             # print("Warning: Could not identify man object or location in state.") # Debug
             return float('inf')


        # 2. Identify usable spanners carried and on the ground
        usable_carried = set()
        usable_on_ground = {} # {spanner_name: location}

        usable_spanners_names = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        for spanner_name in usable_spanners_names:
             is_carried = False
             for fact in state:
                  if match(fact, "carrying", man_name, spanner_name):
                       usable_carried.add(spanner_name)
                       is_carried = True
                       break
             if not is_carried:
                  # Find its location on the ground
                  for fact in state:
                       if match(fact, "at", spanner_name, "*"):
                            usable_on_ground[spanner_name] = get_parts(fact)[2]
                            break

        # 3. Identify loose nuts and their locations
        loose_nuts = {} # {nut_name: location}
        for fact in state:
            if match(fact, "loose", "*"):
                nut_name = get_parts(fact)[1]
                # Find location
                for loc_fact in state:
                    if match(loc_fact, "at", nut_name, "*"):
                        loose_nuts[nut_name] = get_parts(loc_fact)[2]
                        break

        # 4. If no loose nuts, goal reached
        if not loose_nuts:
            return 0

        # 5. Calculate counts
        num_loose = len(loose_nuts)
        num_usable_carried = len(usable_carried)
        num_to_pickup = max(0, num_loose - num_usable_carried)

        # If num_to_pickup > 0 but no usable spanners on ground, unsolvable
        if num_to_pickup > 0 and not usable_on_ground:
             return float('inf')

        # 8. Initialize heuristic cost
        total_heuristic_cost = num_loose + num_to_pickup # Tighten + Pickup actions

        # 9. Estimate travel cost
        # a. Travel to each loose nut location
        for nut_location in loose_nuts.values():
             dist = self.get_distance(man_location, nut_location)
             if dist == float('inf'): return float('inf') # Unreachable nut
             total_heuristic_cost += dist

        # b. Travel to the NumToPickup closest usable spanners on the ground
        if num_to_pickup > 0:
             spanners_by_distance = [] # List of (distance, spanner_name, spanner_location)
             for spanner_name, spanner_location in usable_on_ground.items():
                  dist = self.get_distance(man_location, spanner_location)
                  if dist == float('inf'): continue # Ignore unreachable spanners
                  spanners_by_distance.append((dist, spanner_name, spanner_location))

             # If not enough reachable usable spanners on ground, unsolvable
             if len(spanners_by_distance) < num_to_pickup:
                  return float('inf')

             spanners_by_distance.sort() # Sort by distance
             closest_spanners = spanners_by_distance[:num_to_pickup]

             for dist, _, _ in closest_spanners:
                  total_heuristic_cost += dist

        return total_heuristic_cost
