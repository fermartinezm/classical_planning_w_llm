from fnmatch import fnmatch
import collections

# Helper functions from examples
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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # Check if there are extra parts in fact not covered by args (unless last arg is wildcard)
    if len(parts) > len(args) and args and args[-1] != '*':
         return False
    return True


# Define the heuristic class
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total number of actions required to tighten all loose nuts.
    It sums, for each loose nut, the minimum estimated cost to get a usable spanner
    to the nut's location and perform the tighten action.

    # Assumptions
    - Each loose nut requires a unique usable spanner to be tightened.
    - The cost for each loose nut can be estimated independently, ignoring the
      impact of completing one nut's mission on the man's location and available spanners
      for subsequent nuts (except for counting the total number of usable spanners).
    - The man can only carry one spanner at a time.
    - Spanners become unusable after one use (the 'usable' predicate is removed).
    - Nuts do not change location.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Extracts all location objects and builds a graph representing the links between them.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies all nuts and their initial locations.
    - Identifies which nuts are initially loose (these are the nuts that need tightening).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are currently loose in the state. If none are loose, the heuristic is 0.
    2. Identify the man's current location. If the man's location is unknown, the heuristic is infinity.
    3. Identify all usable spanners (carried or on the ground) and their current locations.
    4. If the number of currently loose nuts exceeds the total number of available usable spanners,
       the problem is likely unsolvable from this state, return a very large value (infinity).
    5. Initialize the total heuristic cost to 0.
    6. For each nut that is currently loose:
       a. Determine the nut's location (which is fixed from the initial state). If the nut's location is unknown, the heuristic is infinity.
       b. Calculate the minimum cost to complete the "mission" for this nut: get a usable spanner
          to the nut's location and perform the tighten action. This minimum is taken over all
          currently available usable spanners (carried or on ground).
          - If considering a usable spanner the man is currently carrying: The cost is the travel distance
            from the man's current location to the nut's location, plus 1 for the tighten action.
            If the nut's location is unreachable from the man's location, this option's cost is infinity.
          - If considering a usable spanner on the ground at location L_S: The cost is the travel distance
            from the man's current location to L_S, plus 1 for the pickup action, plus the travel distance
            from L_S to the nut's location, plus 1 for the tighten action.
            If L_S is unreachable from the man's location, or the nut's location is unreachable from L_S,
            this option's cost is infinity.
          - The minimum of these costs (over all available usable spanners) is the estimated mission cost for this nut.
       c. If the minimum mission cost for this nut is infinity, the state is likely unsolvable, return infinity.
       d. Add this minimum mission cost to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and computing distances."""
        self.task = task # Store task for access to initial state etc.

        # 1. Extract all locations
        self.locations = set()
        # Locations from object definitions typed as 'location'
        self.locations.update(task.objects.get('location', []))

        # Locations from link facts (ensure all linked places are included)
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)

        # 2. Build location graph
        self.graph = collections.defaultdict(list)
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations exist in our set (should be true if parsed correctly)
                if loc1 in self.locations and loc2 in self.locations:
                    self.graph[loc1].append(loc2)
                    self.graph[loc2].append(loc1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 4. Identify initial nut locations (nuts don't move)
        self.initial_nut_locations = {}
        initial_nuts = task.objects.get('nut', [])
        for fact in task.initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in initial_nuts:
                     self.initial_nut_locations[obj] = loc

        # 5. Identify initially loose nuts (these are the nuts that need tightening)
        # This set defines the complete set of nuts that must be tightened to reach the goal.
        self.initially_loose_nuts = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "loose", "*")}

        # Assuming there is exactly one man
        self.man_name = None
        men = self.task.objects.get('man', [])
        if men:
            self.man_name = men[0]


    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances
             return distances # All remain infinity

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Identify current loose nuts
        current_loose_nuts = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}

        # Goal check: If no loose nuts, we are in a goal state.
        if not current_loose_nuts:
            return 0

        # 2. Find man's current location
        man_location = None
        if self.man_name:
             for fact in state:
                 if match(fact, "at", self.man_name, "*"):
                     man_location = get_parts(fact)[2]
                     break

        if man_location is None or man_location not in self.locations:
             # Man's location not found in state, or is in an unknown location
             return float('inf') # Cannot make progress

        # 3. Identify current usable spanners and their locations
        usable_spanners_on_ground = {} # {spanner_name: location}
        carried_usable_spanner = None

        # Find all usable spanners first
        usable_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in usable_spanners_in_state:
                    usable_spanners_on_ground[obj] = loc
            elif self.man_name and match(fact, "carrying", self.man_name, "*"):
                 spanner = get_parts(fact)[2]
                 if spanner in usable_spanners_in_state:
                     carried_usable_spanner = spanner

        # 4. Check solvability based on spanner count
        num_usable_spanners = len(usable_spanners_on_ground) + (1 if carried_usable_spanner is not None else 0)
        if len(current_loose_nuts) > num_usable_spanners:
             # Not enough usable spanners for all loose nuts
             return float('inf')

        # 5. Calculate total heuristic cost
        total_cost = 0
        for nut in current_loose_nuts:
            # Nut location is static, use initial location
            nut_location = self.initial_nut_locations.get(nut)
            if nut_location is None or nut_location not in self.locations:
                 # Nut location not found or is not a known location
                 return float('inf') # Cannot reach nut

            # Calculate min cost to bring a usable spanner to nut_location and tighten
            min_mission_cost = float('inf')

            # Option A: Use carried usable spanner (if any)
            if carried_usable_spanner is not None:
                # Check if man_location can reach nut_location
                if man_location in self.distances and nut_location in self.distances[man_location]:
                     travel_cost = self.distances[man_location][nut_location]
                     if travel_cost != float('inf'):
                         cost_using_carried = travel_cost + 1 # Travel + Tighten
                         min_mission_cost = min(min_mission_cost, cost_using_carried)

            # Option B: Pick up a usable spanner from the ground
            for spanner, spanner_loc in usable_spanners_on_ground.items():
                 # Check if spanner_loc is a known location
                 if spanner_loc not in self.locations:
                      continue # Skip spanners in unknown locations

                 # Check if man_location can reach spanner_loc and spanner_loc can reach nut_location
                 if (man_location in self.distances and spanner_loc in self.distances[man_location] and
                     spanner_loc in self.distances and nut_location in self.distances[spanner_loc]):

                     travel1_cost = self.distances[man_location][spanner_loc]
                     travel2_cost = self.distances[spanner_loc][nut_location]

                     if travel1_cost != float('inf') and travel2_cost != float('inf'):
                         cost_picking_up = travel1_cost + 1 + travel2_cost + 1 # Travel1 + Pickup + Travel2 + Tighten
                         min_mission_cost = min(min_mission_cost, cost_picking_up)

            # If no way (carried or pickup) to get a spanner to this nut, state is likely unsolvable
            if min_mission_cost == float('inf'):
                 return float('inf')

            total_cost += min_mission_cost

        return total_cost
