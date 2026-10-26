from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper function to split a PDDL fact string into parts
def get_parts(fact):
    """Splits a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

# Helper function to match a fact against a pattern
def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        This heuristic estimates the cost to reach a goal state (all required nuts tightened)
        by summing the estimated costs of the necessary actions and travel. It considers
        the number of nuts that need tightening, the number of spanners that need to be
        picked up, and the minimum travel cost to get the man and the first required
        spanner to the location of the first nut to be tightened.

    Assumptions:
        - There is exactly one man object, identifiable (e.g., by name 'bob' or being the single object of 'man' type).
        - Nut locations are static.
        - Spanners are consumed after one use (`tighten_nut` makes them not usable).
        - Links between locations are bidirectional and have a cost of 1.
        - The location graph is connected, or unreachable locations imply unsolvability (handled by returning infinity).
        - Object names follow typical conventions (e.g., contain 'spanner', 'nut', 'location').
        - Initial state and static facts contain sufficient information to identify all relevant objects and their initial/static properties.

    Heuristic Initialization:
        In the __init__ method, the heuristic performs the following precomputations:
        1. Identifies the man, all spanners, all nuts, and all locations present in the problem instance
           by inspecting the initial state and static facts. This relies on simple name-based inference
           and predicate structure.
        2. Stores the set of nuts that are part of the goal condition.
        3. Stores the static location for each nut.
        4. Builds a graph representing the locations and the links between them based on static 'link' facts.
        5. Computes the shortest path distance between all pairs of locations using Breadth-First Search (BFS).
           These distances are stored in a dictionary for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state, the __call__ method computes the heuristic value as follows:
        1. Identify the set of nuts that are currently loose and are also goal conditions.
        2. If this set is empty, the goal is reached, and the heuristic returns 0.
        3. Initialize the heuristic value `h` to 0.
        4. Add the number of loose goal nuts to `h`. This accounts for the minimum number of 'tighten_nut' actions required.
        5. Count the total number of usable spanners in the current state. If this number is less than the number of loose goal nuts, the problem is unsolvable from this state, and the heuristic returns infinity.
        6. Determine the man's current location and whether he is currently carrying a usable spanner. If the man's location cannot be determined, return infinity.
        7. Identify the locations of all usable spanners that are not currently carried by the man.
        8. Identify the locations of all loose nuts that are goals. If any goal nut location is unknown or invalid, return infinity.
        9. Calculate the number of additional spanners the man needs to pick up. This is the total number of loose goal nuts minus one if the man is already carrying a usable spanner (since he can use the one he has for the first nut), capped at zero. Add this number to `h` (each pickup action costs 1).
        10. Estimate the initial travel cost:
            a. If the man is already carrying a usable spanner, he needs to travel from his current location to the closest loose goal nut location. Add this minimum distance to `h`.
            b. If the man is not carrying a usable spanner, he must first travel to an available usable spanner, pick it up, and then travel to the closest loose goal nut location from there. Calculate the minimum cost for this sequence (travel to spanner + travel from spanner to nut) over all available usable spanners. Add this minimum travel cost to `h`. If no available usable spanners or no goal nut locations are reachable, this cost will be infinity, correctly propagating unsolvability.
        11. The final value of `h` is returned. This heuristic is non-admissible as it sums costs that might be achieved concurrently or with shared travel, but it aims to provide a good estimate for greedy best-first search. It will be 0 only in goal states.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object.
        """
        self.goals_set = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Heuristic Initialization ---
        # Identify objects (man, spanners, nuts, locations)
        self.man = None
        self.spanners = []
        self.nuts = []
        self.locations = []

        # Infer objects and types from initial state and static facts
        potential_objects = set()
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if parts: # Ensure fact is not empty or malformed
                 potential_objects.update(parts[1:])

        # Categorize objects based on predicate arguments they appear in
        # This is heuristic, relying on domain structure and naming conventions
        for obj in potential_objects:
            # Check if obj appears as the first argument of 'at' or 'carrying'
            is_locatable_subject = any(match(f, 'at', obj, '*') or match(f, 'carrying', obj, '*') for f in initial_state | static_facts)
            # Check if obj appears as the second argument of 'at' or 'link'
            is_location_object = any(match(f, 'at', '*', obj) or match(f, 'link', obj, '*') or match(f, 'link', '*', obj) for f in initial_state | static_facts)
            # Check if obj appears as the second argument of 'carrying' or argument of 'usable'
            is_spanner_object = any(match(f, 'carrying', '*', obj) or match(f, 'usable', obj) for f in initial_state | static_facts)
            # Check if obj appears as the argument of 'tightened' or 'loose'
            is_nut_object = any(match(f, 'tightened', obj) or match(f, 'loose', obj) for f in initial_state | static_facts)

            # Simple assignment based on typical roles and names
            if is_locatable_subject and not is_spanner_object and not is_nut_object and ('bob' in obj.lower() or 'man' in obj.lower()):
                 self.man = obj
            elif is_spanner_object:
                 self.spanners.append(obj)
            elif is_nut_object:
                 self.nuts.append(obj)
            elif is_location_object:
                 self.locations.append(obj)

        self.spanners = list(set(self.spanners)) # Remove duplicates
        self.nuts = list(set(self.nuts))
        self.locations = list(set(self.locations))

        # Fallback for man identification if needed (e.g., not named 'bob')
        if not self.man:
             # Assume the single object that is a locatable subject but not a spanner or nut is the man
             locatable_subjects = {obj for obj in potential_objects if any(match(f, 'at', obj, '*') or match(f, 'carrying', obj, '*') for f in initial_state | static_facts)}
             other_locatables = locatable_subjects - set(self.spanners) - set(self.nuts)
             if len(other_locatables) == 1:
                  self.man = list(other_locatables)[0]
             # else: print(f"Warning: Could not definitively identify the man object. Candidates: {other_locatables}")


        # Store static nut locations
        self.nut_locations = {}
        for nut in self.nuts:
            found_loc = False
            # Check static facts first
            for fact in static_facts:
                if match(fact, "at", nut, "*"):
                    self.nut_locations[nut] = get_parts(fact)[2]
                    found_loc = True
                    break
            # If not in static, check initial state (should be static for nuts)
            if not found_loc:
                 for fact in initial_state:
                     if match(fact, "at", nut, "*"):
                         self.nut_locations[nut] = get_parts(fact)[2]
                         found_loc = True
                         break
            if not found_loc:
                 # print(f"Warning: Could not find location for nut {nut}")
                 self.nut_locations[nut] = None # Indicate unknown location


        # Build location graph and compute shortest paths (BFS)
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 in self.locations and l2 in self.locations:
                    self.location_graph[l1].add(l2)
                    self.location_graph[l2].add(l1) # Links are bidirectional

        self.dist = {loc: {other_loc: math.inf for other_loc in self.locations} for loc in self.locations}
        for start_node in self.locations:
            self.dist[start_node][start_node] = 0
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                # Ensure u is a valid location key before accessing graph
                if u in self.location_graph:
                    for v in self.location_graph[u]:
                        if self.dist[start_node][v] == math.inf:
                            self.dist[start_node][v] = self.dist[start_node][u] + 1
                            queue.append(v)

        # Store goal nuts for quick lookup
        self.goal_nuts = {get_parts(g)[1] for g in self.goals_set if match(g, "tightened", "*")}


    def get_man_location(self, state):
        """Finds the current location of the man."""
        if not self.man: return None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                return get_parts(fact)[2]
        return None # Man must always be somewhere in a valid state

    def dist(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 is None or loc2 is None or loc1 not in self.locations or loc2 not in self.locations:
             return math.inf
        return self.dist[loc1][loc2]


    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            An estimate of the remaining cost to reach a goal state.
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 1. Identify loose nuts that are goal conditions
        loose_nuts_to_goal = {
            nut for nut in self.goal_nuts
            if f'(loose {nut})' in state # Check if the nut is currently loose
        }
        num_loose_goals = len(loose_nuts_to_goal)

        # If all goal nuts are tightened, the heuristic is 0
        if num_loose_goals == 0:
            return 0

        # Initialize heuristic value
        h = 0

        # 2. Add cost for the 'tighten_nut' action for each loose goal nut
        # Each loose nut that is a goal requires one tighten action.
        h += num_loose_goals

        # 3. Check if enough usable spanners exist
        # Each tighten action consumes a usable spanner.
        usable_spanners_in_state = {s for s in self.spanners if f'(usable {s})' in state}
        num_usable_spanners = len(usable_spanners_in_state)
        if num_loose_goals > num_usable_spanners:
            # Problem is likely unsolvable from this state with available usable spanners
            return math.inf

        # 4. Determine man's current state
        man_loc = self.get_man_location(state)
        if man_loc is None:
             # Man location not found, state is invalid or unexpected
             return math.inf # Should not happen in valid states

        man_carrying_usable = any(
            f'(carrying {self.man} {s})' in state and f'(usable {s})' in state
            for s in self.spanners
        )

        # 5. Identify locations of available usable spanners (not carried by man)
        available_usable_spanners_locs = {
            self.get_parts(fact)[2] for fact in state
            if match(fact, "at", "*", "*") # Find all 'at' facts
            and self.get_parts(fact)[1] in usable_spanners_in_state # Check if the object is a usable spanner
            and f'(carrying {self.man} {self.get_parts(fact)[1]})' not in state # Check man is not carrying it
            and self.get_parts(fact)[2] in self.locations # Ensure location is valid
        }

        # 6. Identify locations of loose nuts that are goals
        nut_goal_locations = {self.nut_locations[nut] for nut in loose_nuts_to_goal if nut in self.nut_locations and self.nut_locations[nut] is not None}

        # If any goal nut location is unknown or invalid, return infinity
        if len(nut_goal_locations) != len(loose_nuts_to_goal):
             return math.inf

        # 7. Add cost for spanner pickups
        # The man needs num_loose_goals spanners in total.
        # He starts with 1 if he is carrying a usable one, otherwise 0.
        # The number of additional spanners he needs to pick up is the difference.
        spanners_to_pickup = max(0, num_loose_goals - (1 if man_carrying_usable else 0))
        h += spanners_to_pickup # Each pickup action costs 1

        # 8. Add cost for initial travel
        # The man needs to get to the first required location (either a spanner to pick up, or a nut location if he has a spanner).
        if man_carrying_usable:
            # Man has a spanner, he can go directly to the closest nut location
            min_dist_to_first_nut = math.inf
            if nut_goal_locations: # Ensure there are nut locations to go to
                 min_dist_to_first_nut = min(self.dist(man_loc, l_n) for l_n in nut_goal_locations)
            h += min_dist_to_first_nut

        else: # Man is not carrying a usable spanner
            # He must first go to an available usable spanner, pick it up, and then go to a nut location.
            min_travel_spanner_then_nut = math.inf
            if available_usable_spanners_locs and nut_goal_locations: # Ensure there are spanners and nuts
                for l_sa in available_usable_spanners_locs:
                    travel_to_spanner = self.dist(man_loc, l_sa)
                    # After picking up at l_sa, he needs to go to the closest nut from there
                    travel_spanner_to_nut = math.inf
                    if nut_goal_locations:
                         travel_spanner_to_nut = min(self.dist(l_sa, l_n) for l_n in nut_goal_locations)

                    # Total cost for this option: travel_to_spanner + travel_spanner_to_nut
                    # Pickup cost is added in step 7.
                    if travel_to_spanner != math.inf and travel_spanner_to_nut != math.inf:
                         min_travel_spanner_then_nut = min(min_travel_spanner_then_nut, travel_to_spanner + travel_spanner_to_nut)

            # If min_travel_spanner_then_nut is still inf, it means no path or no resources.
            # Spanner availability check (step 3) should catch unsolvability based on count.
            # If count is okay but locations are unreachable, this will be inf.
            h += min_travel_spanner_then_nut

        # Note: This heuristic does not explicitly model travel between subsequent nut/spanner locations
        # after the first nut is tightened. The cost of subsequent pickups (step 7) and the base
        # cost of tighten actions (step 2) provide a lower bound, and the initial travel cost
        # (step 8) estimates the effort to start the first task sequence.

        # Handle cases where travel or spanner/nut locations were unreachable/invalid
        if h >= math.inf: # Use >= to catch potential inf + finite = inf
             return math.inf

        return h

    # Provide helper methods used internally
    def get_parts(self, fact):
        return get_parts(fact)

    def match(self, fact, *args):
        return match(fact, *args)
