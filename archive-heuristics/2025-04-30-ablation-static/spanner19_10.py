# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
# If running standalone, define a dummy base class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch
import collections # For BFS queue

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
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
    This heuristic estimates the cost to reach a goal state by considering the number of nuts
    that still need tightening, the distance to the nearest untightened goal nut, and the
    cost to acquire a usable spanner if the man doesn't currently have one. It prioritizes
    reducing the number of loose goal nuts and being in a state where the next tightening
    action (or the steps leading to it) can be performed quickly.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Spanners are consumed after one use (become unusable).
    - The man can carry multiple spanners (inferred from domain structure and examples).
    - Nuts remain at their initial locations throughout the problem.
    - Locations and links form a static graph.
    - The man object can be identified (e.g., by being involved in 'carrying' or 'at' facts).

    # Heuristic Initialization
    - Parses static facts (`link`, initial `at` for nuts) to build the location graph and store static nut locations.
    - Identifies the man's name by looking for the object involved in a 'carrying' fact in the initial state, or falls back based on 'at' facts and naming conventions.
    - Identifies the set of nuts that are goals (need to be tightened) from `task.goals`.
    - Precomputes shortest path distances between all relevant locations using BFS on the location graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Extract the man's current location, the set of usable spanners, and the set of spanners the man is carrying from the state facts.
    2. Identify the set of nuts that are currently loose and are also goal nuts (`loose_goal_nuts`).
    3. If `loose_goal_nuts` is empty, the goal is reached, and the heuristic is 0.
    4. Otherwise, calculate the heuristic value as the sum of three main components:
       a.  **Tightening Cost:** The number of loose goal nuts (`len(loose_goal_nuts)`). This is a lower bound on the number of `tighten_nut` actions required.
       b.  **Approach Nut Cost:** The shortest distance from the man's current location to the location of the nearest nut in `loose_goal_nuts`. This estimates the walk cost to get to where the next tightening action can potentially happen. If no loose goal nuts are reachable, this component contributes infinity.
       c.  **Spanner Acquisition Cost:** An estimate of the cost to ensure the man has a usable spanner *for the immediate next tightening opportunity*.
           - If the man is currently carrying at least one usable spanner, this cost is 0.
           - If the man is carrying one or more spanners, but *none* are usable, this state is penalized with a large value, as the man cannot pick up a new spanner until the unusable one is used (which is impossible if it's unusable) or discarded (no such action).
           - If the man is carrying no spanners, the cost is the shortest distance from the man's current location to any location with a usable spanner on the ground, plus 1 for the `pickup_spanner` action. If no usable spanners are on the ground (and not carried), this component contributes infinity.
    5. The total heuristic value is the sum of components (a), (b), and (c). If any component contributes infinity, the total heuristic is infinity (represented by a large number like 1000000).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        self.location_graph = {}  # Adjacency list for locations {loc: {neighbor1, neighbor2}}
        self.nut_locations = {}  # Map nut -> initial location (static)
        self.man_name = None      # Store the name of the man object

        # Collect all locations mentioned in static facts and initial state
        all_locations_set = set()

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "link" and len(args) == 2:
                l1, l2 = args
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1)
                all_locations_set.add(l1)
                all_locations_set.add(l2)
            elif predicate == "at" and len(args) == 2:
                 obj_name, loc_name = args
                 # Assuming objects starting with 'nut' are nuts based on domain types and examples
                 if obj_name.startswith('nut'):
                     self.nut_locations[obj_name] = loc_name
                 all_locations_set.add(loc_name)

        # Find man's name by looking for the object in a 'carrying' predicate in initial state
        for fact in initial_state:
            if match(fact, "carrying", "*", "*"):
                self.man_name = get_parts(fact)[1]
                break
        # If not found in carrying, look for the object of type 'man' in initial 'at' facts
        # (assuming man is the only 'locatable' that isn't a nut or spanner based on domain types)
        if self.man_name is None:
             locatables_in_init = set()
             spanners_in_init = set()
             nuts_in_init = set()
             for fact in initial_state:
                 parts = get_parts(fact)
                 if not parts: continue
                 if parts[0] == "at" and len(parts) == 2:
                     obj_name, loc_name = parts
                     locatables_in_init.add(obj_name)
                     all_locations_set.add(loc_name)
                 elif parts[0] == "usable" and len(parts) == 1:
                     spanners_in_init.add(parts[1])
                 elif parts[0] == "loose" and len(parts) == 1:
                     nuts_in_init.add(parts[1])

             # The man is a locatable but not a spanner or nut
             man_candidates = locatables_in_init - spanners_in_init - nuts_in_init
             if len(man_candidates) == 1:
                 self.man_name = list(man_candidates)[0]
             elif len(man_candidates) > 1:
                 # Multiple candidates, heuristic might be ambiguous. Fallback to 'bob'.
                 self.man_name = 'bob' # Fallback based on example
             # If len is 0, something is wrong. Man name remains None. Heuristic might fail or return inf.
             # Add initial state locations to the set even if the object type isn't clear yet.
             for fact in initial_state:
                 parts = get_parts(fact)
                 if parts and parts[0] == "at" and len(parts) == 2:
                     all_locations_set.add(parts[2])


        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations_list = list(all_locations_set) # Use a list for consistent iteration order

        for start_loc in all_locations_list:
             self.distances[start_loc] = self.bfs(start_loc, all_locations_set)


    def bfs(self, start_loc, all_locations):
        """
        Perform BFS from start_loc to find distances to all reachable locations.
        """
        distances_from_start = {loc: float('inf') for loc in all_locations}
        if start_loc not in all_locations:
             # Isolated location not in the collected set
             return distances_from_start # All distances remain inf

        distances_from_start[start_loc] = 0
        queue = collections.deque([start_loc])
        visited = {start_loc}

        while queue:
            current_loc = queue.popleft()

            # Check if location has any links in the graph
            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if neighbor in all_locations and neighbor not in visited: # Ensure neighbor is a known location
                        visited.add(neighbor)
                        distances_from_start[neighbor] = distances_from_start[current_loc] + 1
                        queue.append(neighbor)

        return distances_from_start

    def get_distance(self, loc1, loc2):
        """
        Get the precomputed shortest distance between two locations.
        Returns float('inf') if loc1 or loc2 is unknown, or if loc2 is unreachable from loc1.
        """
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
            return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        man_loc = None
        spanner_locations_in_state = {} # Map spanner -> location
        usable_spanners_in_state = set()
        carrying_spanners = set() # Set of spanners the man is carrying

        # Extract relevant information from the current state
        locatables_in_state = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2:
                obj_name, loc_name = args
                locatables_in_state[obj_name] = loc_name
                if obj_name == self.man_name:
                    man_loc = loc_name
            elif predicate == "usable" and len(args) == 1:
                usable_spanners_in_state.add(args[0])
            elif predicate == "carrying" and len(args) == 2 and args[0] == self.man_name:
                carrying_spanners.add(args[1])

        # Identify spanners on the ground based on locatables that are not the man and not known nuts
        # This assumes all other locatables are spanners. This is based on domain types.
        spanners_on_ground = {obj for obj in locatables_in_state if obj != self.man_name and obj not in self.nut_locations}
        spanner_locations_in_state = {s: locatables_in_state[s] for s in spanners_on_ground}


        # Identify loose goal nuts
        loose_goal_nuts = [n for n in self.goal_nuts if f"(loose {n})" in state]

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # --- Heuristic Calculation ---
        h = 0

        # Component 1: Number of tighten actions needed
        h += len(loose_goal_nuts)

        # Component 2: Distance to the nearest loose goal nut
        min_dist_to_nut = float('inf')
        if man_loc: # Only calculate if man's location is known
            for nut_name in loose_goal_nuts:
                # Nut locations are static, use precomputed self.nut_locations
                nut_loc = self.nut_locations.get(nut_name)
                if nut_loc: # Ensure nut location is known (should be from init)
                     dist = self.get_distance(man_loc, nut_loc)
                     min_dist_to_nut = min(min_dist_to_nut, dist)

        # If man_loc is unknown or no loose goal nut location is reachable, return large number
        if min_dist_to_nut == float('inf'):
             return 1000000 # Cannot reach any nut

        h += min_dist_to_nut

        # Component 3: Cost to acquire a usable spanner if needed for the first nut
        spanner_cost = 0
        usable_carried_spanners = carrying_spanners.intersection(usable_spanners_in_state)

        if not usable_carried_spanners: # Man needs a usable spanner
            # Check if man is blocked by carrying an unusable spanner
            if carrying_spanners: # Carrying something, but none are usable
                 spanner_cost = 1000000 # Large penalty: Cannot pick up another
            else: # Carrying nothing
                # Find nearest usable spanner on the ground
                usable_spanners_on_ground_locs = [
                    spanner_locations_in_state[s] for s in usable_spanners_in_state
                    if s in spanner_locations_in_state # Ensure spanner location is known
                ]

                if usable_spanners_on_ground_locs:
                    min_dist_to_spanner = float('inf')
                    if man_loc: # Only calculate if man's location is known
                        for loc_s in usable_spanners_on_ground_locs:
                            dist = self.get_distance(man_loc, loc_s)
                            min_dist_to_spanner = min(min_dist_to_spanner, dist)

                    if min_dist_to_spanner == float('inf'):
                         spanner_cost = 1000000 # Usable spanners on ground exist but are unreachable
                    else:
                         spanner_cost = min_dist_to_spanner + 1 # Walk + pickup
                else:
                    spanner_cost = 1000000 # No usable spanners on the ground

        h += spanner_cost

        # Return large number if any component resulted in infinity
        if h >= 1000000: # Check against the large number used for infinity
             return 1000000

        return h
