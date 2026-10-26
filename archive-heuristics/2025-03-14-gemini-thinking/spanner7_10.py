from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the actions: walk, pickup_spanner, and tighten_nut.

    # Assumptions:
    - The goal is always to tighten all initially loose nuts.
    - The shortest path between locations is a good estimate for walk actions.
    - Picking up the closest usable spanner is always beneficial.

    # Heuristic Initialization
    - Pre-calculates shortest paths between all pairs of locations based on 'link' predicates.
    - Identifies usable spanners and their locations.
    - Identifies initially loose nuts and their locations.
    - Identifies the man and his initial location.

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that is initially loose and not tightened in the current state:
    1. Estimate the cost to reach the nut's location from the man's current location using pre-calculated shortest paths (walk actions).
    2. Check if the man is carrying a usable spanner. If not:
       a. Find the closest location with a usable spanner from the man's current location using pre-calculated shortest paths.
       b. Estimate the cost to reach the closest spanner location (walk actions).
       c. Add 1 action for 'pickup_spanner'.
    3. Add 1 action for 'tighten_nut'.
    Sum up these costs for all initially loose nuts that are not yet tightened in the current state to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the spanner heuristic."""
        self.goals = task.goals
        self.static_facts = task.static

        self.locations = set()
        self.links = {} # Adjacency list for locations
        self.usable_spanner_locations = {} # spanner: location
        self.nut_locations = {} # nut: location
        self.man = None
        self.initial_man_location = None
        self.initial_loose_nuts = set()

        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.links:
                    self.links[loc1] = []
                if loc2 not in self.links:
                    self.links[loc2] = []
                self.links[loc1].append(loc2)
                self.links[loc2].append(loc1)

        for obj in task.objects.values():
            if obj.type_name == 'spanner':
                for fact in task.initial_state:
                    if match(fact, "at", obj.name, "*"):
                        self.usable_spanner_locations[obj.name] = get_parts(fact)[2]
                        break
            elif obj.type_name == 'nut':
                for fact in task.initial_state:
                    if match(fact, "at", obj.name, "*"):
                        self.nut_locations[obj.name] = get_parts(fact)[2]
                        break
                for fact in task.initial_state:
                    if match(fact, "loose", obj.name):
                        self.initial_loose_nuts.add(obj.name)
                        break
            elif obj.type_name == 'man':
                self.man = obj.name
                for fact in task.initial_state:
                    if match(fact, "at", obj.name, "*"):
                        self.initial_man_location = get_parts(fact)[2]
                        break

        self.shortest_paths = self._calculate_shortest_paths()

    def _calculate_shortest_paths(self):
        """Calculates all-pairs shortest paths between locations using BFS."""
        distances = {}
        for start_loc in self.locations:
            distances[start_loc] = {}
            for end_loc in self.locations:
                if start_loc == end_loc:
                    distances[start_loc][end_loc] = 0
                else:
                    distances[start_loc][end_loc] = float('inf')

            queue = [(start_loc, 0)]
            visited = {start_loc}

            while queue:
                current_loc, current_dist = queue.pop(0)
                if current_loc in self.links:
                    for neighbor in self.links[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_loc][neighbor] = current_dist + 1
                            queue.append((neighbor, current_dist + 1))
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        heuristic_cost = 0

        current_man_location = None
        carried_spanner = None
        usable_carried_spanner = False

        for fact in state:
            if match(fact, "at", self.man, "*"):
                current_man_location = get_parts(fact)[2]
            elif match(fact, "carrying", self.man, "*"):
                carried_spanner = get_parts(fact)[2]
                for usable_spanner_name, _ in self.usable_spanner_locations.items():
                    if usable_spanner_name == carried_spanner:
                        usable_carried_spanner = True
                        break


        loose_nuts_to_tighten = []
        for nut in self.initial_loose_nuts:
            goal_tightened_fact = f'(tightened {nut})'
            if goal_tightened_fact not in state:
                loose_nuts_to_tighten.append(nut)

        for nut in loose_nuts_to_tighten:
            nut_location = self.nut_locations[nut]

            # Cost to walk to the nut location
            if current_man_location != nut_location:
                path_cost_to_nut = self.shortest_paths.get(current_man_location, {}).get(nut_location, float('inf'))
                if path_cost_to_nut == float('inf'):
                    return float('inf') # No path to nut, should not happen in valid problems
                heuristic_cost += path_cost_to_nut
                current_man_location = nut_location # Assume man is at nut location after walking (for next nut calculation)


            # Cost to get a usable spanner if not carrying one
            if not usable_carried_spanner:
                min_spanner_path = float('inf')
                closest_spanner_location = None
                for spanner, spanner_loc in self.usable_spanner_locations.items():
                    path_cost_to_spanner = self.shortest_paths.get(current_man_location, {}).get(spanner_loc, float('inf'))
                    if path_cost_to_spanner < min_spanner_path:
                        min_spanner_path = path_cost_to_spanner
                        closest_spanner_location = spanner_loc

                if min_spanner_path == float('inf'):
                    return float('inf') # No usable spanner reachable, should not happen in valid problems

                heuristic_cost += min_spanner_path
                heuristic_cost += 1 # pickup_spanner action
                usable_carried_spanner = True # Assume man is carrying usable spanner after pickup

            heuristic_cost += 1 # tighten_nut action


        return heuristic_cost
