from fnmatch import fnmatch
# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Mock Heuristic base class for standalone testing if needed
# In a real planning environment, this would be provided.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose
    goal nuts. It simulates a greedy process where the man iteratively picks up
    the closest available usable spanner (if needed) and then travels to and
    tightens the closest loose goal nut.

    # Assumptions:
    - The man can carry only one spanner at a time.
    - Each tighten action consumes one usable spanner, making it unusable.
    - Nut locations are static and known from the initial state.
    - All locations mentioned in `link` predicates form a connected graph
      relevant to the problem, or unreachable locations imply infinite cost.
    - The problem is solvable (enough usable spanners exist initially).
      If not enough spanners, heuristic returns infinity.

    # Heuristic Initialization
    - Builds the location graph from static `link` facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies goal nuts and their fixed locations from the initial state.
    - Identifies the man object name from the initial state.

    # Step-By-Step Thinking for Computing Heuristic (__call__)
    1. Identify the man's current location.
    2. Check if the man is currently carrying a usable spanner.
    3. Identify all usable spanners currently available on the ground and their locations.
    4. Identify all goal nuts that are currently loose.
    5. If no goal nuts are loose, the heuristic is 0.
    6. Check if there are enough usable spanners (carried or available) for the remaining loose nuts. If not, return infinity.
    7. Simulate a greedy plan:
       Initialize heuristic cost to 0.
       While there are loose goal nuts:
         a. If the man is carrying a usable spanner:
            Find the loose goal nut closest to the man's current location.
            Add the distance to this nut's location to the heuristic cost.
            Add 1 for the `tighten_nut` action.
            Update the man's current location to the nut's location.
            The spanner is now unusable; the man is no longer carrying a usable spanner.
            Remove the nut from the set of loose nuts.
         b. If the man is not carrying a usable spanner:
            Find the available usable spanner closest to the man's current location.
            Add the distance to this spanner's location to the heuristic cost.
            Add 1 for the `pickup_spanner` action.
            Update the man's current location to the spanner's location.
            The man is now carrying a usable spanner.
            Remove the spanner from the set of available spanners.
    8. Return the total accumulated heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal details.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # 1. Build location graph and compute all-pairs shortest paths
        self.location_graph = {}
        locations = set()
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        self.all_pairs_shortest_paths = {}
        all_locations_list = list(locations) # Ensure consistent order for BFS calls
        for start_loc in all_locations_list:
             self.all_pairs_shortest_paths[start_loc] = self._bfs(start_loc, locations)

        # 2. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "tightened"}

        # 3. Identify spanner and nut objects from initial state based on predicates
        spanners_in_init = {get_parts(fact)[1] for fact in self.initial_state if get_parts(fact)[0] in ["usable", "carrying"]}
        nuts_in_init = {get_parts(fact)[1] for fact in self.initial_state if get_parts(fact)[0] in ["loose", "tightened"]}

        # 4. Identify the man object name
        # The man is the object 'at' a location in the initial state that is not a spanner or nut
        all_objects_at_in_init = {get_parts(fact)[1] for fact in self.initial_state if get_parts(fact)[0] == "at"}
        man_candidates = all_objects_at_in_in

