from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available from the planning framework
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not provided, you might need a dummy definition like this:
# class Heuristic:
#     """Base class for domain-dependent heuristics."""
#     def __init__(self, task):
#         """Initializes the heuristic with the planning task."""
#         self.task = task
#
#     def __call__(self, node):
#         """
#         Computes the heuristic value for a given state node.
#         Returns a non-negative number or float('inf').
#         """
#         raise NotImplementedError("Heuristic subclass must implement __call__")
#
#     def __str__(self):
#         return self.__class__.__name__
#
#     def __repr__(self):
#         return f"<{self.__class__.__name__}>"


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
    # Check if the number of parts is at least the number of arguments in the pattern
    if len(parts) < len(args):
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the number of tightening actions, the number of spanners that need
    to be picked up, and the travel cost to reach the first necessary location
    (either a nut location or a spanner location if pickups are needed).

    # Assumptions
    - Links between locations are unidirectional as defined by the `link` predicate.
    - Each usable spanner can tighten exactly one nut.
    - The man can carry multiple spanners.
    - Nuts stay at their initial locations.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Parses the PDDL task to identify all locations and the man object.
    - Builds a directed graph of locations based on `link` predicates.
    - Computes all-pairs shortest paths between locations using BFS on the directed graph.
    - Identifies the set of nuts that need to be tightened (goal nuts) and their initial locations.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Find the man's current location in the state.
    2. Identify all loose nuts in the current state that are also goal conditions. Count them (`num_nuts_to_tighten`).
    3. If `num_nuts_to_tighten` is 0, the goal is reached, return 0.
    4. Identify usable spanners currently carried by the man. Count them (`num_spanners_carried`).
    5. Identify usable spanners on the ground and their locations. Count them (`num_spanners_ground`) and store their locations (`Spanner_locs`).
    6. If the total number of usable spanners (`num_spanners_carried + num_spanners_ground`) is less than `num_nuts_to_tighten`, the problem is unsolvable from this state, return infinity.
    7. Calculate the number of spanners the man still needs to pick up from the ground (`needed_pickups = max(0, num_nuts_to_tighten - num_spanners_carried)`).
    8. Identify the set of locations of the loose goal nuts (`Nut_locs`).
    9. Determine the set of locations the man needs to reach first. This set includes all `Nut_locs`. If `needed_pickups > 0`, it also includes all `Spanner_locs`.
    10. Calculate the shortest distance from the man's current location to the closest location in the set determined in step 9 (`first_stop_dist`). Use the precomputed distances. If no required location is reachable, `first_stop_dist` will be infinity.
    11. The heuristic estimate is the sum of:
        - The number of tightening actions needed (`num_nuts_to_tighten`).
        - The number of pickup actions needed (`needed_pickups`).
        - The travel cost to reach the first necessary location (`first_stop_dist`).
    12. Return the calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Identify the man object (assuming there's exactly one man)
        self.man_name = None
        # Find the object in the initial state that is involved in an 'at' predicate
        # and is not a spanner or nut based on naming convention, or is involved in 'carrying'.
        potential_men = set()
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 obj_name = get_parts(fact)[1]
                 # Simple check based on naming convention for this domain
                 if not obj_name.startswith("spanner") and not obj_name.startswith("nut"):
                     potential_men.add(obj_name)
             elif match(fact, "carrying", "*", "*"):
                  potential_men.add(get_parts(fact)[1])

        # Assuming there's exactly one man object
        if len(potential_men) == 1:
             self.man_name = potential_men.pop()
        elif len(potential_men) > 1:
             # Handle multiple potential men if necessary, e.g., pick one deterministically
             self.man_name = sorted(list(potential_men))[0] # Example: pick alphabetically first
        else:
             # No potential man found, this might indicate an issue with the problem definition
             self.man_name = None # Heuristic will return infinity if man_name is None


        # 2. Extract all locations and build directed graph
        self.locations = set()
        self.graph = {} # Directed adjacency list {loc: [neighbor1, neighbor2, ...]}

        # Locations from initial state 'at' facts
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                self.locations.add(get_parts(fact)[2])

        # Locations and links from static 'link' facts (directed)
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph.setdefault(l1, []).append(l2)

        # Ensure all locations from graph are in the set
        for loc in self.graph:
             self.locations.add(loc)
             for neighbor in self.graph.get(loc, []): # Use .get for safety
                 self.locations.add(neighbor)

        # 3. Compute all-pairs shortest paths using BFS on the directed graph
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 4. Identify goal nuts
        self.goal_nuts = set()
        # Goal facts are like (tightened nut1)
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

        # 5. Identify initial nut locations (nuts don't move)
        self.initial_nut_locations = {}
        for fact in self.initial_state:
            if match(fact, "at", "*", "*") and get_parts(fact)[1] in self.goal_nuts:
                 self.initial_nut_locations[get_parts(fact)[1]] = get_parts(fact)[2]


    def _bfs(self, start_node):
        """
        Performs BFS from a start node on the directed graph to find shortest distances.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_node in distances: # Handle case where start_node might not be in self.locations
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current = queue.popleft()

                # Handle locations that might be in self.locations but not in self.graph (e.g., isolated locations)
                if current not in self.graph:
                     continue

                for neighbor in self.graph[current]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Retrieves the precomputed shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This indicates an issue, likely loc1 or loc2 is not a known location
             # or there's no path. Return infinity.
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Handle case where man_name was not identified during initialization
        if self.man_name is None:
             return float('inf')

        # 1. Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man is not at any location? Problem state is invalid for this heuristic.
             return float('inf')

        # Ensure man's current location is one we know about from initialization
        if man_location not in self.locations:
             # This state contains a location not seen in init/static facts. Invalid state for this heuristic.
             return float('inf')


        # 2. Identify loose goal nuts in the current state and their locations
        loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}
        num_nuts_to_tighten = len(loose_goal_nuts)

        # 3. If num_nuts_to_tighten is 0, the goal is reached
        if num_nuts_to_tighten == 0:
            return 0

        # 4. Identify usable spanners (carried and on ground)
        carried_spanners = {get_parts(fact)[2] for fact in state if match(fact, "carrying", self.man_name, "*")}
        usable_spanners_carried = {s for s in carried_spanners if f"(usable {s})" in state}
        num_spanners_carried = len(usable_spanners_carried)

        spanners_on_ground = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj_name = get_parts(fact)[1]
                 # Check if it's a spanner (assuming naming convention)
                 if obj_name.startswith("spanner"):
                     spanners_on_ground[obj_name] = get_parts(fact)[2]

        usable_spanners_ground_locations = {
            loc for spanner, loc in spanners_on_ground.items() if f"(usable {spanner})" in state
        }
        num_spanners_ground = len(usable_spanners_ground_locations)

        # 5. Check for unsolvable state (not enough usable spanners in total)
        if num_nuts_to_tighten > num_spanners_carried + num_spanners_ground:
            return float('inf')

        # 6. Calculate needed pickups
        needed_pickups = max(0, num_nuts_to_tighten - num_spanners_carried)

        # 7. Identify required locations
        nut_locations = {self.initial_nut_locations[nut] for nut in loose_goal_nuts if nut in self.initial_nut_locations}
        spanner_locations = usable_spanners_ground_locations

        # 8. Calculate distance to the first necessary location
        # This is the minimum distance from the man's current location
        # to any location he *must* visit first.
        # If pickups are needed, he might go to a spanner location first.
        # He must eventually go to nut locations.
        # The first stop is the closest reachable location among all required nut locations
        # and potentially required spanner pickup locations.

        required_first_visit_locations = set()
        if num_nuts_to_tighten > 0:
             required_first_visit_locations.update(nut_locations)
        if needed_pickups > 0:
             required_first_visit_locations.update(spanner_locations)

        first_stop_dist = float('inf')
        if man_location in self.distances: # Ensure man's location is in our precomputed distances
            for loc in required_first_visit_locations:
                 dist = self.get_distance(man_location, loc)
                 if dist != float('inf'):
                     first_stop_dist = min(first_stop_dist, dist)

        # If after checking all required locations, first_stop_dist is still infinity,
        # it means none of the required locations are reachable from the man's current location.
        # This implies the problem is unsolvable from this state.
        if first_stop_dist == float('inf') and num_nuts_to_tighten > 0:
             return float('inf')
        elif first_stop_dist == float('inf') and num_nuts_to_tighten == 0:
             # This case should be caught earlier, but as a fallback
             first_stop_dist = 0 # Goal reached, no travel needed

        # 9. Calculate heuristic value
        # H = (tighten actions) + (pickup actions) + (initial travel)
        heuristic_value = num_nuts_to_tighten + needed_pickups + first_stop_dist

        return heuristic_value
