from fnmatch import fnmatch
import collections # For BFS queue

# Assuming Heuristic base class is available in heuristics.heuristic_base
# If not, a dummy base class is needed for the code to be runnable standalone.
# In a real planning framework, this import would be sufficient:
# from heuristics.heuristic_base import Heuristic

# Define a dummy base class if the actual one is not provided in the execution environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """
        Dummy base class for domain-dependent heuristics.
        A heuristic must implement __init__(self, task) and __call__(self, node).
        """
        def __init__(self, task):
            """
            Initializes the heuristic with the planning task.
            :param task: An instance of the Task class.
            """
            self.task = task

        def __call__(self, node):
            """
            Computes the heuristic value for a given state node.
            :param node: A search node containing the state.
            :return: An estimated cost to reach the goal from the node's state.
            """
            raise NotImplementedError("Heuristic subclass must implement __call__")


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
    # Handle trailing wildcard specifically
    if args and args[-1] == '*':
         if len(parts) < len(args) - 1: return False
         return all(fnmatch(part, arg) for part, arg in zip(parts[:len(args)-1], args[:-1]))

    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
         return False

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose
    goal nuts. It sums the estimated cost for each individual loose goal nut,
    considering the cost to move the man to the nut's location with a usable
    spanner, plus the tighten action.

    # Assumptions
    - There is exactly one man.
    - Nut locations are static.
    - Spanners are consumed after one use (become unusable).
    - The man can carry at most one spanner at a time.
    - The location graph defined by 'link' predicates is undirected (man can walk both ways).
    - All necessary usable spanners exist in the initial state for solvable problems.

    # Heuristic Initialization
    - Identify the single man object.
    - Identify all location objects.
    - Store the static locations of all nuts.
    - Store the set of nuts that need to be tightened (goal nuts).
    - Build the location graph based on 'link' facts, assuming undirected edges.
    - Compute all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Determine if the man is currently carrying a usable spanner.
    3. Identify the locations of all usable spanners that are currently on the ground.
    4. Identify the set of loose nuts that are also goal nuts. If this set is empty, the heuristic is 0.
    5. Initialize the total heuristic cost to 0.
    6. For each loose goal nut 'n' at its static location 'L_n':
        a. Calculate the estimated cost to get the man to 'L_n' carrying a usable spanner.
           - If the man is currently carrying a usable spanner: This cost is the shortest distance from the man's current location to 'L_n'.
           - If the man is not carrying a usable spanner (or carrying an unusable one): This cost is the minimum over all locations 'L_s' with a usable spanner on the ground, of the path cost: distance(man's location, L_s) + 1 (pickup action) + distance(L_s, L_n).
           - If the man needs a spanner but no usable spanners are available on the ground, this cost is considered infinite (or a very large number), indicating an unsolvable state for this nut.
        b. Add the calculated cost from step 6a, plus 1 (for the 'tighten_nut' action), to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        super().__init__(task) # Call base class constructor
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Identify objects and their types based on initial state and static facts
        all_facts = set(self.initial_state) | set(self.static_facts)

        potential_men = {get_parts(fact)[1] for fact in all_facts if match(fact, 'carrying', '*', '*')}
        potential_spanners = {get_parts(fact)[2] for fact in all_facts if match(fact, 'carrying', '*', '*')} | \
                             {get_parts(fact)[1] for fact in all_facts if match(fact, 'usable', '*')}
        potential_nuts = {get_parts(fact)[1] for fact in all_facts if match(fact, 'tightened', '*')} | \
                         {get_parts(fact)[1] for fact in all_facts if match(fact, 'loose', '*')}
        potential_locations = set()
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] == 'link':
                  potential_locations.add(parts[1])
                  potential_locations.add(parts[2])
             elif parts[0] == 'at':
                  potential_locations.add(parts[2]) # The second argument of 'at' is always a location

        # Assuming there is exactly one man
        if len(potential_men) == 1:
             self.man = list(potential_men)[0]
        else:
             # Fallback: Find the object in (at ...) that is not a spanner or nut in the initial state
             all_locatables_at_start = {get_parts(fact)[1] for fact in self.initial_state if match(fact, 'at', '*', '*')}
             man_candidates_fallback = all_locatables_at_start - potential_spanners - potential_nuts
             if len(man_candidates_fallback) == 1:
                  self.man = list(man_candidates_fallback)[0]
             else:
                  # Cannot reliably identify the single man. Heuristic might be broken.
                  # This case indicates the domain structure might differ from assumptions
                  # or the provided facts are insufficient for unique identification.
                  # For this problem, we proceed assuming a single man is identifiable
                  # by one of the above methods or is the first locatable in initial state
                  # if no 'carrying' facts exist. Let's pick the first locatable as a last resort.
                  if all_locatables_at_start:
                       self.man = list(all_locatables_at_start)[0]
                  else:
                       self.man = None # Cannot identify man at all


        self.spanners = potential_spanners
        self.nuts = potential_nuts
        self.locations = potential_locations


        # Store static locations of nuts
        self.nut_locations = {}
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj in self.nuts:
                    self.nut_locations[obj] = loc
                # Note: Man and spanners also have initial locations, but they change.
                # Nut locations are assumed static based on domain structure (no action moves nuts).


        # Store goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build location graph (assuming undirected links)
        self.location_graph = collections.defaultdict(list)
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                # Ensure locations are identified before adding to graph
                if l1 in self.locations and l2 in self.locations:
                    self.location_graph[l1].append(l2)
                    self.location_graph[l2].append(l1) # Assume undirected

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all reachable locations."""
        distances_from_start = {loc: float('inf') for loc in self.locations}
        if start_loc not in self.locations:
             # Start location is not a recognized location object
             return distances_from_start # All distances remain infinity

        distances_from_start[start_loc] = 0
        queue = collections.deque([start_loc])

        while queue:
            curr_loc = queue.popleft()

            # Ensure curr_loc is in graph keys before accessing
            if curr_loc in self.location_graph:
                for neighbor in self.location_graph[curr_loc]:
                    # Ensure neighbor is a recognized location object
                    if neighbor in self.locations and distances_from_start[neighbor] == float('inf'):
                        distances_from_start[neighbor] = distances_from_start[curr_loc] + 1
                        queue.append(neighbor)
        return distances_from_start

    def get_distance(self, loc1, loc2):
        """Looks up precomputed distance, returns infinity if unreachable or locations invalid."""
        if loc1 not in self.distances or loc2 not in self.locations:
             return float('inf')
        return self.distances[loc1].get(loc2, float('inf'))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if man was identified in __init__
        if self.man is None:
             # Cannot compute heuristic if man object is unknown
             return float('inf') # Indicate unsolvable/error state

        # Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f"(loose {n})" in state}

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None or man_location not in self.locations:
             # Man must always be at a known location in a valid state
             return float('inf') # Should not happen in solvable problems

        # Determine if man is carrying a usable spanner
        man_carrying_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man, "*"):
                man_carrying_spanner = get_parts(fact)[2]
                break

        man_carrying_usable_spanner = (man_carrying_spanner is not None) and (f"(usable {man_carrying_spanner})" in state)

        # Identify usable spanners on the ground
        usable_spanner_locations = set()
        spanners_on_ground = set()
        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                  if obj in self.spanners:
                       spanners_on_ground.add(obj)

        for spanner in spanners_on_ground:
             if f"(usable {spanner})" in state:
                  # Find location of this usable spanner on the ground
                  for fact in state:
                       if match(fact, "at", spanner, "*"):
                            usable_spanner_locations.add(get_parts(fact)[2])
                            break # Found location, move to next spanner


        # Initialize total heuristic cost
        total_cost = 0
        LARGE_NUMBER = 1000000 # Use a large number for unreachable/unsolvable parts

        # For each loose goal nut
        for nut in loose_goal_nuts:
            nut_location = self.nut_locations.get(nut)
            if nut_location is None or nut_location not in self.locations:
                 # Nut location not found or not a recognized location
                 return LARGE_NUMBER # Indicate unsolvable/error state

            # Calculate cost to get man+usable_spanner to nut_location
            cost_to_get_man_spanner_to_nut_loc = float('inf')

            if man_carrying_usable_spanner:
                # Man brings the spanner with him
                dist = self.get_distance(man_location, nut_location)
                if dist != float('inf'):
                    cost_to_get_man_spanner_to_nut_loc = dist
            else:
                # Man needs to pick up a spanner first
                if not usable_spanner_locations:
                    # No usable spanners available to pick up
                    cost_to_get_man_spanner_to_nut_loc = float('inf') # Cannot get a spanner
                else:
                    min_spanner_path_cost = float('inf')
                    for spanner_loc in usable_spanner_locations:
                        dist_to_spanner = self.get_distance(man_location, spanner_loc)
                        if dist_to_spanner != float('inf'):
                            # Cost to go to spanner, pickup, then go to nut
                            dist_spanner_to_nut = self.get_distance(spanner_loc, nut_location)
                            if dist_spanner_to_nut != float('inf'):
                                path_cost = dist_to_spanner + 1 + dist_spanner_to_nut
                                min_spanner_path_cost = min(min_spanner_path_cost, path_cost)
                    cost_to_get_man_spanner_to_nut_loc = min_spanner_path_cost

            # Add cost for this nut
            if cost_to_get_man_spanner_to_nut_loc == float('inf'):
                 # This nut is unreachable or requires an unavailable spanner
                 return LARGE_NUMBER # Problem is likely unsolvable from here

            total_cost += cost_to_get_man_spanner_to_nut_loc + 1 # +1 for the tighten action

        # Return total heuristic cost
        # Cap the heuristic value to avoid potential overflow or issues with very large sums
        # A large number like 1000000 indicates practical unsolvability within reasonable search limits
        return min(total_cost, LARGE_NUMBER)
