import collections
from fnmatch import fnmatch
# Assuming Heuristic base class is available in a module named heuristics
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe raise an error or return None/empty list
        return [] # Return empty list for invalid format
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Placeholder for the base Heuristic class if not provided externally
# In a real setup, this would be imported.
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError("Heuristic must implement __call__")


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to reach the goal state (all required nuts
    tightened) by summing the estimated costs for each individual loose nut
    that needs to be tightened. The cost for a single nut is estimated based on
    the actions required to get the man to the nut's location with a usable
    spanner and then tightening the nut. It uses shortest path distances on the
    location graph to estimate movement costs.

    # Assumptions
    - The location graph defined by 'link' predicates is static and connected
      among relevant locations for solvable problems.
    - Shortest path distances between locations represent the minimum number of
      'walk' actions.
    - A spanner becomes unusable after tightening one nut.
    - If the man is not carrying a usable spanner, he will pick up the closest
      available usable spanner.
    - There is always at least one usable spanner available somewhere if needed
      in a solvable problem instance.
    - The cost of each action (walk, pickup, tighten) is 1.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Identify the man object from the initial state (assuming it's the object
      involved in 'carrying' or the unique 'locatable' that isn't a spanner/nut).
    - Extract the goal conditions (`tightened` predicates for specific nuts).
    - Build the location graph from the static `link` predicates.
    - Compute all-pairs shortest path distances between all locations using BFS.
      This precomputation allows quick lookup of travel costs during heuristic
      evaluation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Identify all nuts that are in the goal state (`(tightened ?n)`) but are
       currently `loose` (i.e., `(tightened ?n)` is not in the current state).
       These are the "loose goal nuts".
    3. For each loose goal nut `N`:
       a. Cost for this nut starts at 1 (representing the `tighten_nut` action).
       b. Find the current location of the man (`L_M`) and the location of the
          nut (`L_N`) from the state.
       c. Determine if the man is currently carrying a usable spanner.
          - Find all spanners `S` the man is carrying (`(carrying man S)`).
          - Check if any of these carried spanners `S` are `usable` (`(usable S)`).
       d. If the man is carrying a usable spanner:
          - The cost to get the man (with spanner) to the nut's location is the
            shortest path distance `distance(L_M, L_N)`. Add this distance to
            the cost for this nut.
       e. If the man is NOT carrying a usable spanner:
          - Find all usable spanners `S_avail` that are currently at some
            location `L_S` (`(at S_avail L_S)` and `(usable S_avail)`).
          - If there are no such usable spanners available at locations, the
            problem might be unsolvable from this state. Return infinity (or a
            very large number) as the heuristic value.
          - Otherwise, the man must go to a spanner, pick it up, and then go
            to the nut. The path is `L_M -> L_S -> L_N`. The cost for this part
            is `distance(L_M, L_S) + 1 (pickup) + distance(L_S, L_N)`.
          - Find the usable spanner `S` at location `L_S` that minimizes this
            `distance(L_M, L_S) + distance(L_S, L_N)`.
          - Add the minimum of `distance(L_M, L_S) + 1 + distance(L_S, L_N)`
            over all available usable spanners to the cost for this nut.
       f. Add the total cost calculated for this nut to the overall `total_cost`.
    4. Return the `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and computing distances."""
        super().__init__(task)

        # Find the man object. Assume it's the object in a 'carrying' fact
        # or the unique object of type 'man' if types were available.
        # Based on examples, 'bob' is the man. Let's try to find it dynamically.
        self.man_obj = None
        # Look for an object involved in a 'carrying' fact in the initial state
        for fact in self.task.initial_state:
            if match(fact, "carrying", "*", "*"):
                self.man_obj = get_parts(fact)[1]
                break
        # If no 'carrying' fact in initial state, look for a single object
        # that is 'at' a location but is not a spanner or nut (fragile)
        # A more robust way would require parsing object types from the PDDL problem.
        # Given the constraints, let's assume 'bob' is the man or find the unique
        # object in initial state that is 'at' a location and is not a spanner/nut.
        if self.man_obj is None:
             potential_men = set()
             potential_spanners = set()
             potential_nuts = set()
             for fact in self.task.initial_state:
                 if match(fact, "at", "*", "*"):
                     obj, _ = get_parts(fact)[1], get_parts(fact)[2]
                     # Simple check based on example naming conventions
                     if not obj.startswith('spanner') and not obj.startswith('nut'):
                         potential_men.add(obj)
                 elif match(fact, "usable", "*"):
                     potential_spanners.add(get_parts(fact)[1])
                 elif match(fact, "loose", "*"):
                     potential_nuts.add(get_parts(fact)[1])

             # Remove spanners and nuts from potential men (double check)
             potential_men = potential_men - potential_spanners - potential_nuts

             if len(potential_men) == 1:
                 self.man_obj = list(potential_men)[0]
             elif len(potential_men) > 1:
                 # Ambiguous man object, heuristic might be unreliable
                 # print("Warning: Multiple potential man objects found. Heuristic might be inaccurate.")
                 self.man_obj = list(potential_men)[0] # Pick one arbitrarily
             else:
                 # No man object found? Problem might be malformed.
                 # print("Error: No man object identified in initial state.")
                 self.man_obj = "unknown_man" # Use a placeholder


        # Build the location graph from static link facts
        self.graph = collections.defaultdict(set)
        self.locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Links are bidirectional
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not in the graph, cannot reach anything
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Handle nodes with no links (isolated locations)
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Estimate the minimum cost to tighten all loose goal nuts."""
        state = node.state

        # Extract relevant information from the current state
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
        spanners_at_loc = {} # {spanner_name: location}
        nut_locations = {} # {nut_name: location}
        # all_spanners = set() # Not strictly needed for heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj == self.man_obj:
                     man_location = loc
                elif obj.startswith('nut'): # Simple check based on example names
                     nut_locations[obj] = loc
                elif obj.startswith('spanner'): # Simple check based on example names
                     spanners_at_loc[obj] = loc
                     # all_spanners.add(obj)

            elif predicate == "carrying":
                 carrier, spanner_obj = parts[1], parts[2]
                 if carrier == self.man_obj:
                     carried_spanners.add(spanner_obj)
                     # all_spanners.add(spanner_obj)

            elif predicate == "usable":
                 spanner_obj = parts[1]
                 usable_spanners.add(spanner_obj)


        # Check if man location was found
        if man_location is None:
             # Man's location must be known. Should not happen in valid states.
             return float('inf')

        # Identify usable spanners currently carried by the man
        carried_usable_spanners = carried_spanners.intersection(usable_spanners)

        # Identify usable spanners currently at locations
        available_usable_spanners_at_loc = {
            s: loc for s, loc in spanners_at_loc.items() if s in usable_spanners
        }

        total_cost = 0

        # Iterate through goal conditions to find loose goal nuts
        loose_goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                # Check if the nut is NOT yet tightened in the current state
                if goal not in state:
                    loose_goal_nuts.add(nut)

        # Calculate cost for each loose goal nut
        for nut in loose_goal_nuts:
            # Cost starts with the tighten action
            nut_cost = 1

            # Find the nut's current location
            L_N = nut_locations.get(nut)
            if L_N is None:
                 # Nut location must be known. Should not happen in valid states.
                 return float('inf') # Nut disappeared?

            # Check if man_location and L_N are valid locations in our distance map
            if man_location not in self.distances or L_N not in self.distances.get(man_location, {}):
                 # Man or nut is at an unknown or disconnected location
                 return float('inf')


            # Cost to get a usable spanner and get to the nut location
            if carried_usable_spanners:
                # Man is already carrying a usable spanner.
                # Cost is just moving the man (with spanner) to the nut location.
                movement_cost = self.distances[man_location][L_N]
                if movement_cost == float('inf'): return float('inf') # Cannot reach nut
                nut_cost += movement_cost
            else:
                # Man needs to acquire a usable spanner.
                # He must go to a spanner location L_S, pick it up, then go to L_N.
                # The path is L_M -> L_S -> L_N.
                # Cost is distance(L_M, L_S) + 1 (pickup) + distance(L_S, L_N).
                # We need to find the usable spanner S at L_S that minimizes this total travel+pickup cost.

                best_spanner_acquisition_movement_cost = float('inf')

                if not available_usable_spanners_at_loc:
                    # No usable spanners available at locations.
                    # Since no usable spanner is carried either, this is likely unsolvable.
                    return float('inf')

                # Find the best usable spanner to pick up
                for s, L_S in available_usable_spanners_at_loc.items():
                    # Check if L_S is a valid location in our distance map
                    if L_S not in self.distances.get(man_location, {}) or L_N not in self.distances.get(L_S, {}):
                         # Cannot reach spanner or reach nut from spanner location
                         continue # Skip this spanner

                    # Cost to go from man's current location to spanner, pick up, then go to nut
                    dist_m_s = self.distances[man_location][L_S]
                    dist_s_n = self.distances[L_S][L_N]

                    if dist_m_s == float('inf') or dist_s_n == float('inf'):
                         continue # Cannot reach spanner or nut

                    spanner_acquisition_movement_cost = dist_m_s + 1 + dist_s_n
                    best_spanner_acquisition_movement_cost = min(best_spanner_acquisition_movement_cost, spanner_acquisition_movement_cost)

                if best_spanner_acquisition_movement_cost == float('inf'):
                     # Could not find any usable spanner that allows reaching the nut
                     return float('inf')

                nut_cost += best_spanner_acquisition_movement_cost

            total_cost += nut_cost

        return total_cost
