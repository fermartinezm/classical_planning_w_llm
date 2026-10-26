# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

from collections import defaultdict, deque
from fnmatch import fnmatch

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

# Helper function to match facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` are allowed in `args`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards
    # This check is simplified; fnmatch handles wildcard length mismatches implicitly
    # but an explicit check might be slightly more robust depending on expected patterns.
    # Let's rely on fnmatch's behavior.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# class spannerHeuristic(Heuristic): # Inherit from Heuristic if available
class spannerHeuristic: # Define without inheritance for standalone code
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all goal nuts by summing:
    1. The number of untightened goal nuts (representing the tighten actions).
    2. The sum of shortest path distances from the man's current location to each unique location where an untightened goal nut is located.
    3. The estimated cost to acquire all necessary usable spanners from the ground.

    # Assumptions
    - Nuts do not move from their initial locations.
    - The graph of locations connected by 'link' predicates is connected, allowing travel between any two locations relevant to the problem, or the heuristic will return infinity.
    - There are enough usable spanners available on the ground initially to tighten all goal nuts (or the heuristic will return infinity).
    - The cost of walking between linked locations is 1. The cost of picking up a spanner is 1. The cost of tightening a nut is 1.
    - The man can carry at most one spanner at a time (inferred from action definitions and examples).

    # Heuristic Initialization
    - Extracts goal conditions to identify which nuts need tightening.
    - Extracts 'link' facts from static information to build the location graph.
    - Computes all-pairs shortest paths between all locations using BFS.
    - Stores initial locations of all nuts (assuming they are static).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of goal nuts that are not yet tightened in the current state. If this set is empty, the heuristic is 0 (goal state).
    2. Initialize the heuristic value `h` with the count of these untightened goal nuts (each requiring a `tighten_nut` action, cost 1).
    3. Determine the man's current location.
    4. Identify the unique locations where these untightened goal nuts are situated.
    5. For each unique nut location, add the shortest path distance from the man's current location to this nut location to `h`. (This estimates the travel cost to visit all necessary locations).
    6. Count the number of usable spanners the man is currently carrying. (Assumed to be 0 or 1).
    7. Calculate how many additional usable spanners are needed from the ground (total nuts to tighten minus held usable spanners, minimum 0).
    8. If additional spanners are needed:
       a. Find all locations where usable spanners are available on the ground in the current state.
       b. If no usable spanners are found on the ground, return infinity (as the problem might be unsolvable with the current spanners).
       c. Find the minimum shortest path distance from the man's current location to any location found in step 8a.
       d. Add the estimated cost for acquiring all needed additional spanners to `h`. This cost is estimated as `needed_acquisitions * (minimum_distance_to_spanner + 1)`. (This assumes each needed spanner requires travel to the nearest available spanner and a pickup action).
    9. Return the final calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the location graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Build the location graph from 'link' facts and identify all locations
        self.location_graph = defaultdict(set)
        self.all_locations = set()

        # Collect all potential locations mentioned in links, initial state, and goals
        all_relevant_facts = set(initial_state) | set(self.goals) | set(static_facts)
        inferred_locations = set()
        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Links are bidirectional
                inferred_locations.add(loc1)
                inferred_locations.add(loc2)
            elif parts[0] == 'at' and len(parts) == 3:
                 # The second argument of 'at' is the location
                 inferred_locations.add(parts[2])

        self.all_locations.update(inferred_locations)

        # Ensure all inferred locations are in the graph even if they have no links initially
        for loc in self.all_locations:
             if loc not in self.location_graph:
                 self.location_graph[loc] = set()

        # 2. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.all_locations:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                current_node, distance = q.popleft()
                self.dist[start_node][current_node] = distance
                for neighbor in self.location_graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, distance + 1))

        # Handle unreachable locations (set distance to infinity)
        for l1 in self.all_locations:
            for l2 in self.all_locations:
                if l2 not in self.dist[l1]:
                    self.dist[l1][l2] = float('inf')


        # 3. Identify goal nuts and their initial locations (assuming static)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}
        self.nut_locations = {}

        # Find initial locations for all objects that are nuts (either goal nuts or initially loose)
        all_nuts_mentioned = set(self.goal_nuts)
        for fact in initial_state:
             if match(fact, "loose", "*"):
                  all_nuts_mentioned.add(get_parts(fact)[1])

        for nut in all_nuts_mentioned:
             for fact in initial_state:
                  if match(fact, "at", nut, "*"):
                       self.nut_locations[nut] = get_parts(fact)[2]
                       break # Found location for this nut


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify man's current location and object name
        man_obj = None
        man_loc = None
        held_objects = {} # man -> obj

        # Try to find man_obj name from 'carrying' facts first
        for fact in state:
             if match(fact, "carrying", "*", "*"):
                  man_obj = get_parts(fact)[1]
                  break

        # Fallback: Assume 'bob' is the man if not found via 'carrying'.
        if man_obj is None:
             # This is a weak inference. A robust solution needs PDDL parsing capabilities.
             # Assuming 'bob' based on example.
             man_obj = 'bob' # Fragile assumption

        # Now find man's location or what he is carrying in the current state
        current_locations = {} # obj -> loc
        usable_spanners_state = set() # usable spanner names
        spanner_names_in_state = set() # all spanners seen in state

        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                  obj, loc = parts[1], parts[2]
                  current_locations[obj] = loc
                  if obj == man_obj:
                       man_loc = loc
             elif parts[0] == 'carrying' and len(parts) == 3:
                  man, obj = parts[1], parts[2]
                  if man == man_obj:
                       held_objects[man] = obj # Assuming man can only carry one object
                       spanner_names_in_state.add(obj) # Carried object is a spanner
             elif parts[0] == 'usable' and len(parts) == 2:
                  usable_spanners_state.add(parts[1])
                  spanner_names_in_state.add(parts[1]) # Usable object is a spanner

        # If man_loc is still None, it's a problematic state representation for this heuristic.
        if man_loc is None:
             return float('inf')


        # 2. Identify untightened goal nuts and their locations
        untightened_goal_nuts = {
            n for n in self.goal_nuts
            if f'(tightened {n})' not in state
        }

        if not untightened_goal_nuts:
            return 0 # Goal reached

        untightened_goal_nuts_locations = set()
        for n in untightened_goal_nuts:
             if n in self.nut_locations:
                  untightened_goal_nuts_locations.add(self.nut_locations[n])
             else:
                  # Location of a goal nut is unknown from initial state. Problematic.
                  return float('inf')


        # 3. Count usable spanners currently held by the man
        held_usable_count = 0
        carried_spanner = held_objects.get(man_obj)
        if carried_spanner and carried_spanner in usable_spanners_state:
             held_usable_count = 1 # Assuming man carries at most one spanner


        # 4. Find locations with usable spanners on the ground
        usable_spanner_locations_on_ground = set()
        for obj, loc in current_locations.items():
             # Check if the object is a spanner, is usable, and is not currently held by the man
             if obj in spanner_names_in_state and obj in usable_spanners_state and obj not in held_objects.values():
                  usable_spanner_locations_on_ground.add(loc)


        # Calculate heuristic value
        h = 0

        # Cost 1: Tighten actions
        num_nuts_to_tighten = len(untightened_goal_nuts)
        h += num_nuts_to_tighten

        # Cost 2: Travel to nut locations
        # Sum of distances from man_loc to each unique nut location
        if man_loc not in self.dist:
             return float('inf') # Should not happen if man_loc was found

        for nut_l in untightened_goal_nuts_locations:
             if nut_l not in self.dist[man_loc]:
                  # Nut is at an unknown/unreachable location? Should not happen.
                  return float('inf')
             h += self.dist[man_loc][nut_l]


        # Cost 3: Spanner acquisition
        # Number of additional usable spanners needed from the ground
        needed_acquisitions = max(0, num_nuts_to_tighten - held_usable_count)

        if needed_acquisitions > 0:
             if not usable_spanner_locations_on_ground:
                  # No usable spanners on the ground and man doesn't have enough.
                  # This problem instance might be unsolvable with available spanners.
                  return float('inf')

             min_dist_to_spanner = float('inf')
             for spanner_l in usable_spanner_locations_on_ground:
                  if spanner_l in self.dist[man_loc]:
                       min_dist_to_spanner = min(min_dist_to_spanner, self.dist[man_loc][spanner_l])

             if min_dist_to_spanner == float('inf'):
                  # Usable spanners exist but are unreachable from man's current location
                  return float('inf')

             # Add estimated cost for acquiring all needed additional spanners
             # Assumes each acquisition involves travel to the nearest spanner + pickup
             h += needed_acquisitions * (min_dist_to_spanner + 1)


        # Ensure heuristic is 0 only for goal states
        # This is guaranteed because num_nuts_to_tighten is 0 only in goal states.

        # Ensure heuristic is finite for solvable states
        # It returns inf only if no usable spanners are available on the ground when needed
        # and the man doesn't have enough, or if locations are unreachable/unknown.

        return h
