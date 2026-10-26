# Import necessary modules
from collections import deque
from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not available for standalone testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Helper to parse a PDDL fact string into a list of parts."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal state
    in the Spanner domain. It is designed for greedy best-first search and is
    not admissible. The estimate is based on the number of remaining loose goal
    nuts, the number of spanners that need to be acquired, and an estimate of
    the movement cost for the man to reach relevant locations.

    Assumptions:
    - There is exactly one man object in the domain.
    - Nuts do not change location. Their initial location is their fixed location.
    - Spanners do not change location unless carried by the man.
    - The location graph defined by 'link' facts is generally connected for all
      relevant locations (man's start, nut locations, spanner locations) within
      a solvable problem instance. Unreachable locations will result in an
      infinite heuristic value.
    - The goal only consists of `(tightened ?n)` facts for specific nuts.
    - The internal state representation uses strings like `'(predicate arg1 arg2)'`.
    - The `heuristics.heuristic_base.Heuristic` class provides the necessary base structure.

    Heuristic Initialization:
    The heuristic is initialized once per planning task. During initialization,
    it performs the following steps:
    1. Parses object names (man, spanners, nuts, locations) by examining all
       facts in the initial state, static facts, and goal conditions.
    2. Stores the initial locations of all nuts, assuming they are static.
    3. Builds the graph of locations based on the `(link ?l1 ?l2)` static facts.
       The graph is represented as an adjacency list.
    4. Computes all-pairs shortest paths between all known locations using
       Breadth-First Search (BFS). These distances are stored in a dictionary
       `self.dist`, where `self.dist[l1][l2]` is the shortest distance
       (number of walk actions) from location `l1` to location `l2`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        initial_state = task.initial_state
        static_facts = task.static

        # --- Parse Objects and Initial Nut Locations ---
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.location_names = set()
        self.nut_locations = {} # nut -> initial location

        # Collect all objects and locations mentioned in initial state, static, and goals
        all_relevant_facts = set(initial_state) | set(static_facts) | set(self.goals)

        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any
            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.location_names.add(loc)
            elif pred == 'carrying' and len(parts) == 3:
                man, spanner = parts[1], parts[2]
                self.man_name = man # Assume one man
                self.spanner_names.add(spanner)
            elif pred == 'usable' and len(parts) == 2:
                spanner = parts[1]
                self.spanner_names.add(spanner)
            elif pred == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.location_names.add(l1)
                self.location_names.add(l2)
            elif (pred == 'tightened' or pred == 'loose') and len(parts) == 2:
                nut = parts[1]
                self.nut_names.add(nut)

        # Store initial locations of nuts (nuts are static)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj in self.nut_names:
                     self.nut_locations[obj] = loc

        # --- Build Location Graph and Compute Distances ---
        self.links = {l: set() for l in self.location_names}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 in self.links and l2 in self.links: # Ensure locations are known
                    self.links[l1].add(l2)
                    self.links[l2].add(l1)

        self.dist = {}
        for start_loc in self.location_names:
            self.dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all reachable locations."""
        q = deque([(start_loc, 0)])
        visited = {start_loc}
        distances = {start_loc: 0}

        while q:
            curr, d = q.popleft()
            # Check if curr is a valid key in self.links
            if curr in self.links:
                for neighbor in self.links[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = d + 1
                        q.append((neighbor, d + 1))
        return distances

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current state and the goal conditions.
        2. Find the man's current location by iterating through the state facts.
        3. Identify which goal nuts are currently loose. These are the goal nuts
           for which the fact `(tightened nut_name)` is not present in the state.
        4. Count the number of loose goal nuts (`N_loose_goals`). If this count is 0,
           it means all goal nuts are tightened, so the goal is reached and the
           heuristic value is 0.
        5. Identify all usable spanners in the current state by checking for the
           `(usable ?s)` fact.
        6. Count the number of usable spanners currently carried by the man by
           checking for `(carrying man_name ?s)` facts where `?s` is a usable spanner.
           This count is `N_usable_carried`.
        7. Identify usable spanners that are currently located at some location
           (i.e., not carried by the man).
        8. Determine the locations where these usable spanners (from step 7) are
           located. These are the `AvailableSpannerLocations`.
        9. Calculate the total number of usable spanners available in the state,
           which is the sum of carried usable spanners and usable spanners at locations.
           This count is `N_usable_total`.
        10. Check for unsolvability: If the number of loose goal nuts (`N_loose_goals`)
            is greater than the total number of usable spanners available (`N_usable_total`),
            the problem is impossible to solve from this state (as each tightening
            consumes a spanner). In this case, return `float('inf')`.
        11. The base heuristic cost is the number of loose goal nuts (`N_loose_goals`).
            This represents the minimum number of `tighten_nut` actions required.
        12. Calculate the number of additional usable spanners the man needs to acquire
            from locations. This is `N_needed_spanners = max(0, N_loose_goals - N_usable_carried)`.
            Each needed spanner requires at least one `pickup_spanner` action, so add
            `N_needed_spanners` to the heuristic cost.
        13. Estimate the movement cost: The man needs to move from his current location
            to relevant locations to perform actions. Relevant locations are those
            of loose goal nuts (`NutLocations`) and those with available usable spanners
            (`AvailableSpannerLocations`).
            - Identify the locations of loose goal nuts (`NutLocations`).
            - Calculate the minimum distance from the man's current location (`ManLoc`)
              to any loose nut location (`min_dist_to_nut`).
            - Calculate the minimum distance from the man's current location (`ManLoc`)
              to any available usable spanner location (`min_dist_to_spanner`).
            - The movement cost is estimated based on what the man needs to do next:
                - If the man is already carrying enough usable spanners (`N_usable_carried >= N_loose_goals`),
                  he can proceed directly to tighten nuts. The estimated movement cost is the
                  distance to the closest loose nut location (`min_dist_to_nut`).
                - If the man needs more spanners (`N_usable_carried < N_loose_goals`):
                    - If he is carrying *some* usable spanners (`N_usable_carried > 0`), he
                      could potentially go to a nut first (using a carried spanner) or go
                      get more spanners. The estimated movement cost is the minimum of the
                      distance to the closest loose nut and the distance to the closest
                      available spanner location (`min(min_dist_to_nut, min_dist_to_spanner)`).
                    - If he is carrying *no* usable spanners (`N_usable_carried == 0`), he
                      *must* go get a spanner first before tightening any nut. The estimated
                      movement cost is the distance to the closest available usable spanner
                      location (`min_dist_to_spanner`).
            - If the calculated movement cost is `float('inf')`, it means the necessary
              locations are unreachable from the man's current location, so the problem
              is unsolvable from this state. Return `float('inf')`.
        14. Sum the base cost (`N_loose_goals`), the pickup cost (`N_needed_spanners`),
            and the estimated movement cost to get the final heuristic value.
        15. Ensure the final heuristic value is non-negative.

        """
        state = node.state
        goals = self.goals

        # 3. Identify loose goal nuts
        LooseGoalNuts = {get_parts(goal)[1] for goal in goals if match(goal, "tightened", "*") and (goal not in state)}
        # 4. Count loose goal nuts and check if goal is reached
        N_loose_goals = len(LooseGoalNuts)

        if N_loose_goals == 0:
            return 0

        # 2. Find man's current location
        ManLoc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                ManLoc = get_parts(fact)[2]
                break
        # Should not happen in a valid state, but handle defensively
        if ManLoc is None or ManLoc not in self.dist:
             # Man's location is unknown or not in the precomputed graph
             return float('inf')

        # 5. Identify usable spanners in state
        UsableSpannersInState = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # 6. Count usable spanners carried by man
        N_usable_carried = sum(1 for fact in state if match(fact, "carrying", self.man_name, "*") and get_parts(fact)[2] in UsableSpannersInState)

        # 7-8. Identify usable spanners at locations and their locations
        UsableSpannersAtLoc = set()
        AvailableSpannerLocations = set()
        for s in UsableSpannersInState:
            # Check if spanner 's' is carried by the man
            is_carried = False
            for fact in state:
                if match(fact, "carrying", self.man_name, s):
                    is_carried = True
                    break
            # If not carried, check if it's at a location
            if not is_carried:
                for fact in state:
                    if match(fact, "at", s, "*"):
                        AvailableSpannerLocations.add(get_parts(fact)[2])
                        UsableSpannersAtLoc.add(s) # Keep track of spanner names too if needed later
                        break # A spanner is only at one location at a time

        # 9. Total usable spanners
        N_usable_total = N_usable_carried + len(UsableSpannersAtLoc)

        # 10. Unsolvability check
        if N_loose_goals > N_usable_total:
            return float('inf')

        # 11. Base cost (tighten actions)
        h = N_loose_goals

        # 12. Pickup cost
        N_needed_spanners = max(0, N_loose_goals - N_usable_carried)
        h += N_needed_spanners

        # 13. Movement cost
        NutLocations = {self.nut_locations[n] for n in LooseGoalNuts if n in self.nut_locations} # Ensure nut location is known

        min_dist_to_nut = float('inf')
        if NutLocations:
            # Get distances from ManLoc, handle unreachable locations
            distances_from_man = self.dist.get(ManLoc, {})
            min_dist_to_nut = min((distances_from_man.get(l, float('inf')) for l in NutLocations), default=float('inf'))

        min_dist_to_spanner = float('inf')
        if AvailableSpannerLocations:
             # Get distances from ManLoc, handle unreachable locations
            distances_from_man = self.dist.get(ManLoc, {})
            min_dist_to_spanner = min((distances_from_man.get(l, float('inf')) for l in AvailableSpannerLocations), default=float('inf'))

        movement_cost = 0
        # Only add movement cost if there are goals to achieve and relevant locations exist
        if N_loose_goals > 0:
            if N_usable_carried >= N_loose_goals:
                # Enough spanners, go to closest nut
                movement_cost = min_dist_to_nut
            else: # N_usable_carried < N_loose_goals
                if N_usable_carried > 0:
                    # Has some spanners, might go to nut or get more
                    # Need both nuts and spanners to be reachable
                    if min_dist_to_nut == float('inf') and min_dist_to_spanner == float('inf'):
                         return float('inf') # Cannot reach any relevant location
                    movement_cost = min(min_dist_to_nut, min_dist_to_spanner)
                else: # N_usable_carried == 0
                    # Needs spanners, must go get one first
                    if min_dist_to_spanner == float('inf'):
                         return float('inf') # Cannot reach any spanner
                    movement_cost = min_dist_to_spanner

        # If movement_cost ended up being infinity because a required target set was empty
        # (e.g., N_loose_goals > 0 but NutLocations is empty, or N_needed_spanners > 0 but AvailableSpannerLocations is empty)
        # or because all target locations were unreachable from ManLoc, return infinity.
        # The check `if movement_cost == float('inf'): return float('inf')` after the if block is simpler.
        if movement_cost == float('inf'):
             return float('inf')


        h += movement_cost

        # 15. Ensure heuristic is non-negative (should be guaranteed by logic, but safe)
        return max(0, h)
