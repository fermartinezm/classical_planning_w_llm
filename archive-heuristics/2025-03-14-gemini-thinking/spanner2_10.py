from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

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
    This heuristic estimates the minimum number of actions required to tighten all loose nuts.
    It considers the actions: walk, pickup_spanner, and tighten_nut. For each loose nut, it estimates
    the cost to reach the nut's location, acquire a usable spanner if needed, and then tighten the nut.

    # Assumptions:
    - The problem is solvable, meaning there exists a path between locations and usable spanners are reachable.
    - The cost of each action (walk, pickup_spanner, tighten_nut) is 1.
    - We prioritize using a spanner at the current location if available, otherwise, we find the closest usable spanner.

    # Heuristic Initialization
    - Extracts static information about location links to calculate shortest paths.
    - Identifies all nuts that are initially loose and need to be tightened according to the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that is loose in the current state and needs to be tightened according to the goal:
    1. Check if the nut is already tightened in the current state. If yes, the cost for this nut is 0.
    2. If the nut is still loose:
       a. Initialize the cost for this nut to 1 (for the 'tighten_nut' action).
       b. Determine the current location of the man and the nut.
       c. If the man is not at the nut's location, calculate the shortest path (number of 'walk' actions)
          to reach the nut's location using the 'link' predicates. Add this path length to the cost.
       d. Check if the man is carrying a usable spanner.
       e. If not carrying a usable spanner:
          i. Check if there is a usable spanner at the man's current location. If yes, add 1 to the cost
             (for 'pickup_spanner' action).
          ii. If not, find the location of the closest usable spanner using shortest path calculation.
              Add the path length (number of 'walk' actions) to reach the spanner's location and 1
              (for 'pickup_spanner' action) to the cost.
    3. Sum up the costs calculated for each loose nut to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts (links between locations)
        and identifying the goal nuts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.links = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.links[loc1].append(loc2)
                self.links[loc2].append(loc1) # Assuming links are bidirectional for simplicity in path finding

        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])


    def __call__(self, node):
        """
        Calculate the heuristic value for the given state.
        """
        state = node.state
        heuristic_cost = 0

        goal_nut_names = self.goal_nuts

        current_loose_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                current_loose_nuts.add(get_parts(fact)[1])

        nuts_to_tighten = goal_nut_names.intersection(current_loose_nuts)

        if not nuts_to_tighten:
            goal_achieved = True
            for goal_nut in goal_nut_names:
                if not f'(tightened {goal_nut})' in state:
                    goal_achieved = False
                    break
            if goal_achieved:
                return 0


        man_location = None
        carrying_spanner = None
        usable_spanners_locations = {}
        nut_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                loc = parts[2]
                if match(fact, "at", "?m - man", "*"):
                    man_location = loc
                elif match(fact, "at", "?s - spanner", "*"):
                    usable = False
                    for usable_fact in state:
                        if match(usable_fact, "usable", parts[1]):
                            usable = True
                            break
                    if usable:
                        usable_spanners_locations[obj] = loc
                elif match(fact, "at", "?n - nut", "*"):
                    nut_locations[obj] = loc
            elif match(fact, "carrying", "*", "*"):
                carrying_spanner = get_parts(fact)[2]


        for nut_name in nuts_to_tighten:
            nut_cost = 1 # for tighten_nut action
            nut_loc = nut_locations[nut_name]

            if man_location != nut_loc:
                path_len = self._get_shortest_path_len(man_location, nut_loc)
                if path_len is None:
                    return float('inf') # Should not happen in well-formed problems, but handle for robustness
                nut_cost += path_len

            if carrying_spanner is None or not f'(usable {carrying_spanner})' in state:
                spanner_at_location = None
                for spanner, loc in usable_spanners_locations.items():
                    if loc == man_location:
                        spanner_at_location = spanner
                        break
                if spanner_at_location:
                    nut_cost += 1 # pickup_spanner action
                else:
                    closest_spanner_loc_info = self._find_closest_usable_spanner_location(man_location, usable_spanners_locations)
                    if closest_spanner_loc_info:
                        spanner_loc, path_len = closest_spanner_loc_info
                        nut_cost += path_len + 1 # walk to spanner + pickup_spanner
                    else:
                        return float('inf') # No usable spanner reachable, should not happen in solvable problems

            heuristic_cost += nut_cost

        return heuristic_cost

    def _get_shortest_path_len(self, start_loc, end_loc):
        """
        Calculates the shortest path length between two locations using BFS.
        Returns path length or None if no path exists.
        """
        if start_loc == end_loc:
            return 0

        queue = collections.deque([(start_loc, 0)]) # (location, distance)
        visited = {start_loc}

        while queue:
            current_loc, distance = queue.popleft()

            for neighbor in self.links.get(current_loc, []):
                if neighbor not in visited:
                    if neighbor == end_loc:
                        return distance + 1
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None # No path found

    def _find_closest_usable_spanner_location(self, current_location, usable_spanners_locations):
        """
        Finds the closest location with a usable spanner from the current location.
        Returns a tuple: (spanner_location, path_length) or None if no usable spanner is reachable.
        """
        closest_loc = None
        min_path_len = float('inf')

        for spanner, spanner_loc in usable_spanners_locations.items():
            path_len = self._get_shortest_path_len(current_location, spanner_loc)
            if path_len is not None and path_len < min_path_len:
                min_path_len = path_len
                closest_loc = spanner_loc

        if closest_loc is not None:
            return closest_loc, min_path_len
        return None


class Heuristic: # dummy Heuristic class for import context
    pass
