from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
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
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the number of nuts remaining, the need for a usable spanner,
    and the travel cost for the man to reach the locations of the loose nuts.

    # Assumptions
    - The goal is to tighten all nuts defined in the problem's goal state.
    - Spanners become unusable after one use for tightening a nut.
    - The man can carry multiple spanners.
    - The problem instances are solvable with the given number of initial usable spanners.
    - Travel cost between linked locations is 1.
    - The man object is named 'bob'.

    # Heuristic Initialization
    - Extracts all location names and the links between them from static facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies all nuts that need to be tightened from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location. Assume the man object is named 'bob'.
    2. Identify all nuts that are currently loose. A nut is loose if the fact `(loose <nut-name>)` is in the state and the nut is one of the nuts specified in the goal to be tightened.
    3. For each loose nut, find its current location.
    4. Identify all usable spanners currently on the ground and their locations. A spanner is usable if `(usable <spanner-name>)` is in the state and it is on the ground (`(at <spanner-name> <location-name>)` is in the state).
    5. Count the number of usable spanners the man is currently carrying. A spanner is carried if `(carrying bob <spanner-name>)` is in the state, and it is usable if `(usable <spanner-name>)` is also in the state.
    6. If there are no loose nuts, the heuristic is 0, as the goal is reached.
    7. Check if the total number of usable spanners available in the state (carried + on ground) is less than the number of loose nuts. If so, it's impossible to tighten all nuts with the current resources, so return infinity.
    8. Initialize the heuristic value `h` with the number of loose nuts. This is the minimum number of `tighten_nut` actions required.
    9. Calculate the estimated cost to acquire usable spanners if the man is not currently carrying enough for all loose nuts. If the number of loose nuts is greater than the number of usable spanners the man is carrying, he needs to pick up additional spanners from the ground. We estimate the cost to pick up *at least one* of these needed spanners by finding the nearest usable spanner on the ground to the man's current location and adding its distance plus the pickup action cost (1). This is a simplification for multiple spanner acquisition.
    10. Calculate the estimated movement cost for the man to reach the locations where loose nuts are situated. This is estimated by summing the shortest distances from the man's current location to each *unique* location that contains one or more loose nuts. This is a non-admissible estimate for visiting multiple locations but provides a greedy direction.
    11. The total heuristic value is the sum of the base cost (number of loose nuts), the estimated spanner acquisition cost (if needed), and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying goal nuts.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify all locations and build the graph
        self.locations = set()
        self.adjacency_list = {}

        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adjacency_list.setdefault(loc1, set()).add(loc2)
                self.adjacency_list.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        # Ensure all locations mentioned in other static facts (like initial object locations) are included
        for fact in static_facts:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.locations.add(loc)
                 self.adjacency_list.setdefault(loc, set()) # Add location even if it has no links

        # 2. Compute all-pairs shortest paths using BFS
        self.distance = {}
        for start_node in self.locations:
            self.distance[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()
                self.distance[start_node][current_loc] = dist

                # Handle locations with no links explicitly (though BFS naturally stops)
                if current_loc not in self.adjacency_list:
                     continue

                for neighbor in self.adjacency_list[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Ensure all locations are in the distance map, marking unreachable ones
        for loc1 in self.locations:
             for loc2 in self.locations:
                  if loc2 not in self.distance[loc1]:
                       # Assume unreachable locations have infinite distance
                       self.distance[loc1][loc2] = float('inf')


        # 3. Identify all nuts that need tightening from goal facts
        self.all_nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.all_nuts_to_tighten.add(nut)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location (assuming man is 'bob')
        man_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                 man_location = get_parts(fact)[2]
                 break

        if man_location is None:
             # Man's location not found, problem state is likely malformed or unsolvable
             return float('inf')

        # 2. Identify loose nuts and their locations
        loose_nuts = set()
        nut_loc = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in self.all_nuts_to_tighten: # Only care about nuts that need tightening
                    loose_nuts.add(nut)

        # Get locations for loose nuts
        for nut in loose_nuts:
             for fact in state:
                  if match(fact, "at", nut, "*"):
                       nut_loc[nut] = get_parts(fact)[2]
                       break # Found location for this nut
             if nut not in nut_loc:
                  # Loose nut location not found, state is inconsistent
                  return float('inf')


        # 3. Identify usable spanners on the ground and their locations
        usable_spanners_on_ground = set()
        usable_spanner_loc = {}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # Check if this usable spanner is on the ground
                for loc_fact in state:
                    if match(loc_fact, "at", spanner, "*"):
                        location = get_parts(loc_fact)[2]
                        usable_spanners_on_ground.add(spanner)
                        usable_spanner_loc[spanner] = location
                        break # Found location for this spanner

        # 4. Count usable spanners the man is carrying (assuming man is 'bob')
        num_carrying_usable = 0
        carried_spanners = set()
        for fact in state:
             if match(fact, "carrying", "bob", "*"):
                  carried_spanners.add(get_parts(fact)[2])

        for spanner in carried_spanners:
             if f"(usable {spanner})" in state:
                  num_carrying_usable += 1


        # 5. If no loose nuts, return 0
        if not loose_nuts:
            return 0

        # 6. Check solvability based on total usable spanners
        total_usable_spanners = num_carrying_usable + len(usable_spanners_on_ground)
        if len(loose_nuts) > total_usable_spanners:
             # Not enough usable spanners in the entire state to tighten all loose nuts
             return float('inf') # Indicate unsolvable

        # 7. Initialize heuristic with base cost (tighten actions)
        h = len(loose_nuts)

        # 8. Calculate spanner acquisition cost if needed
        # The man needs len(loose_nuts) usable spanners in total.
        # He currently carries num_carrying_usable usable spanners.
        # He needs to acquire max(0, len(loose_nuts) - num_carrying_usable) more usable spanners.
        needed_spanners_to_carry = max(0, len(loose_nuts) - num_carrying_usable)

        if needed_spanners_to_carry > 0:
            usable_on_ground_locs = {l for s, l in usable_spanner_loc.items()}
            # If needed_spanners_to_carry > 0 and total_usable_spanners >= len(loose_nuts),
            # there must be usable spanners on the ground.
            min_dist_to_spanner = float('inf')
            for loc in usable_on_ground_locs:
                 if man_location in self.distance and loc in self.distance[man_location]:
                      min_dist_to_spanner = min(min_dist_to_spanner, self.distance[man_location][loc])

            if min_dist_to_spanner == float('inf'):
                 # This case should ideally be caught by the total_usable_spanners check,
                 # but serves as a safeguard if a usable spanner on ground is unreachable.
                 return float('inf')

            # Estimate cost to get the first needed spanner: walk to nearest + pickup
            h += min_dist_to_spanner + 1

        # 9. Calculate movement cost to nuts
        loose_nut_locations = set(nut_loc.values())

        if loose_nut_locations:
            # Estimate movement cost by summing distances from man's location to each unique nut location.
            movement_cost = 0
            for loc in loose_nut_locations:
                 if man_location in self.distance and loc in self.distance[man_location]:
                      movement_cost += self.distance[man_location][loc]
                 else:
                      # Nut location unreachable, should not happen in solvable problems
                      return float('inf')
            h += movement_cost

        # 10. Return heuristic value
        return h
