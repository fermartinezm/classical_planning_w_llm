import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_loc, graph, locations):
    """
    Perform Breadth-First Search to find shortest distances from a start location
    to all other locations in the graph.

    Args:
        start_loc (str): The starting location.
        graph (dict): Adjacency list representation of the location graph.
                      {location: [neighbor1, neighbor2, ...]}
        locations (set): A set of all location names.

    Returns:
        dict: A dictionary mapping each location to its shortest distance from start_loc.
              Returns float('inf') for unreachable locations.
    """
    distances = {loc: float('inf') for loc in locations}
    distances[start_loc] = 0
    queue = collections.deque([(start_loc, 0)])

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc in graph: # Ensure current_loc is a valid key in graph
            for neighbor in graph[current_loc]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all loose nuts. It sums the
    estimated minimum cost for each loose nut independently. The estimated cost
    for a single nut is the cost of the man traveling to a usable spanner,
    picking it up, traveling to the nut's location, and tightening the nut.
    It finds the minimum such cost over all available usable spanners for each nut.

    # Assumptions:
    - The goal is to tighten all nuts that are initially loose.
    - Each tighten action requires a *usable* spanner, and makes that spanner unusable.
    - Travel between linked locations costs 1 action. Pickup costs 1, Tighten costs 1.
    - The heuristic calculates the cost for each loose nut independently, assuming
      the man starts from his current location for each nut's subproblem. This
      makes it non-admissible but potentially effective for greedy search.

    # Heuristic Initialization
    - Builds a graph of locations based on `link` predicates.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the man object.

    # Step-By-Step Thinking for Computing Heuristic (__call__)
    1. Identify the man's current location.
    2. Identify all nuts that are currently loose and their locations.
    3. Identify all spanners that are currently usable and their locations
       (either on the ground or carried by the man).
    4. If the number of loose nuts exceeds the number of usable spanners, the
       goal is unreachable in this domain (spanners are not made usable again),
       so return infinity.
    5. Initialize total heuristic cost to 0.
    6. For each loose nut:
       a. Determine the nut's location.
       b. Find the minimum cost to get a *usable* spanner to this nut's location,
          carried by the man, starting from the man's current location.
          - Iterate through all currently usable spanners.
          - For a usable spanner `s` at location `l_s` (or carried):
            - Cost = Distance(man_loc, l_s) + (1 if spanner is on ground else 0 for pickup) + Distance(l_s, nut_loc).
          - The minimum of these costs over all usable spanners is the minimum spanner delivery cost for this nut.
       c. Add the minimum spanner delivery cost + 1 (for the tighten action) to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing
        distances between locations.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find the man object

        # 1. Build the location graph from link predicates
        self.locations = set()
        graph = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                graph[loc1].append(loc2)
                graph[loc2].append(loc1) # Assuming links are bidirectional
                self.locations.add(loc1)
                self.locations.add(loc2)

        # 2. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(start_loc, graph, self.locations)

        # 3. Identify the man object (assuming there's only one)
        self.man = None
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj_type = None
                # Need to find the type of the object to confirm it's the man
                # PDDL parsing usually provides object types, but it's not directly
                # available in the 'Task' object provided. A common way is to look
                # for the object definition in the problem file, but we only have
                # the initial state facts here.
                # A robust way would be to parse the objects section.
                # For this heuristic, let's assume the object starting with 'bob'
                # or similar typical man names is the man, or look for the object
                # that is 'carrying' or is the first argument of 'at' in the initial
                # state that isn't a spanner or nut.
                # A simpler, potentially brittle way: find the object that is 'at'
                # a location in the initial state and is not a spanner or nut
                # mentioned in the goals/initial state facts for spanners/nuts.
                # Let's find the object that is 'at' a location and is the first
                # argument of any 'carrying' predicate in the initial state.
                # If no 'carrying' initially, just find the one 'at' a location
                # that isn't a spanner or nut.

                # Find all initial spanners and nuts to exclude them
                initial_spanners = {get_parts(f)[1] for f in initial_state if match(f, "usable", "*")}
                initial_nuts = {get_parts(f)[1] for f in initial_state if match(f, "loose", "*")}

                # Find the object at a location that is not a spanner or nut
                predicate, obj, loc = get_parts(fact)
                if predicate == "at" and obj not in initial_spanners and obj not in initial_nuts:
                     self.man = obj
                     break # Assuming only one man

        if not self.man:
             # Fallback: If no man found this way, maybe look for the first object
             # that is 'at' a location. This is less safe but might work for simple problems.
             for fact in initial_state:
                 if match(fact, "at", "*", "*"):
                      _, obj, _ = get_parts(fact)
                      self.man = obj
                      break
             # If still no man, the heuristic might fail or return inf.
             if not self.man:
                 print("Warning: Could not identify the man object in spannerHeuristic.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                _, _, man_loc = get_parts(fact)
                break
        if man_loc is None:
             # Man is not at any location? Should not happen in valid states.
             # Or maybe the man object wasn't found in __init__ correctly.
             # Return infinity if man location is unknown.
             return float('inf')


        # 2. Find current loose nuts and their locations
        loose_nuts = {} # {nut_name: location}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # Find location of this nut
                for loc_fact in state:
                    if match(loc_fact, "at", nut, "*"):
                        _, _, nut_loc = get_parts(loc_fact)
                        loose_nuts[nut] = nut_loc
                        break

        # 3. Find current usable spanners and their status/locations
        usable_spanners = {} # {spanner_name: {'loc': location, 'carried': bool}}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # Check if carried
                is_carried = False
                for carry_fact in state:
                    if match(carry_fact, "carrying", self.man, spanner):
                        is_carried = True
                        break

                if is_carried:
                    usable_spanners[spanner] = {'loc': man_loc, 'carried': True}
                else:
                    # Find location if not carried
                    spanner_loc = None
                    for loc_fact in state:
                        if match(loc_fact, "at", spanner, "*"):
                            _, _, spanner_loc = get_parts(loc_fact)
                            break
                    if spanner_loc: # Spanner might be unusable and not at a location if dropped somewhere not linked? Or maybe unusable spanners disappear? Assuming usable ones are either at a location or carried.
                         usable_spanners[spanner] = {'loc': spanner_loc, 'carried': False}
                    # If a usable spanner is not carried and not at a location, something is wrong with the state or domain model interpretation. We'll ignore it for the heuristic.


        # 4. Check if goal is reachable based on spanners
        num_loose_nuts = len(loose_nuts)
        num_usable_spanners = len(usable_spanners)

        if num_loose_nuts == 0:
            return 0 # Goal reached

        if num_usable_spanners < num_loose_nuts:
            # Not enough usable spanners to tighten all loose nuts
            return float('inf')

        # 5. Calculate total heuristic cost
        total_cost = 0

        for nut, nut_loc in loose_nuts.items():
            min_cost_for_this_nut = float('inf')

            # Find the minimum cost to get a usable spanner to nut_loc and tighten it
            for spanner, spanner_info in usable_spanners.items():
                spanner_loc = spanner_info['loc']
                is_carried = spanner_info['carried']

                # Cost to get man from current location to spanner location
                # Handle cases where man_loc or spanner_loc might not be in distances (e.g., initial state locations not linked)
                if man_loc not in self.distances or spanner_loc not in self.distances[man_loc]:
                     # This shouldn't happen if all locations are linked or part of the graph
                     travel_to_spanner = float('inf')
                else:
                     travel_to_spanner = self.distances[man_loc][spanner_loc]

                # Cost to pick up spanner (0 if carried, 1 if on ground)
                pickup_cost = 0 if is_carried else 1

                # Cost to get man (with spanner) from spanner location to nut location
                if spanner_loc not in self.distances or nut_loc not in self.distances[spanner_loc]:
                     # This shouldn't happen if all locations are linked or part of the graph
                     travel_spanner_to_nut = float('inf')
                else:
                     travel_spanner_to_nut = self.distances[spanner_loc][nut_loc]

                # Cost to tighten nut
                tighten_cost = 1

                # Total cost for this nut using this spanner (independent subproblem)
                cost_for_this_spanner = travel_to_spanner + pickup_cost + travel_spanner_to_nut + tighten_cost

                min_cost_for_this_nut = min(min_cost_for_this_nut, cost_for_this_spanner)

            # If min_cost_for_this_nut is still infinity, it means no usable spanner could reach this nut location
            # from the man's current location via known links. This state might be a dead end.
            # However, the check num_usable_spanners < num_loose_nuts should catch the main unreachability.
            # If there are usable spanners but none are reachable from man_loc, this sum will become inf.
            total_cost += min_cost_for_this_nut

        return total_cost

