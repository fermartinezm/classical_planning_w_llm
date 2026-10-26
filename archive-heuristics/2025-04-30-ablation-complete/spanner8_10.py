from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class is available in the environment and imported elsewhere
# e.g., from heuristics.heuristic_base import Heuristic

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
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Inherit from Heuristic if the base class is available in the environment
# class spannerHeuristic(Heuristic):
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It calculates shortest path distances between locations using BFS.
    The heuristic is the sum, over all loose goal nuts, of the estimated cost to tighten that specific nut,
    assuming the man starts from his current state. This cost includes travel for the man and a spanner,
    plus the pickup and tighten actions.

    # Assumptions
    - Spanners are single-use (become unusable after tightening one nut).
    - A man can only carry one spanner at a time.
    - Nuts are static (their location does not change).
    - Links between locations are bidirectional for distance calculation.
    - Enough usable spanners exist to tighten all goal nuts (checked by heuristic).
    - Object names follow conventions (e.g., 'nut' for nuts, 'spanner' for spanners, man is the object involved in 'carrying' or the only non-nut/spanner locatable).

    # Heuristic Initialization
    - Identify all locations from initial state and static facts (`link`).
    - Build the location graph from `link` facts (assuming bidirectionality).
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify all nut objects that are goals and their static locations from the initial state.
    - Store the goal conditions (which nuts need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of nuts that are currently loose but need to be tightened according to the goal. Let this set be N_loose_goal.
    2. If N_loose_goal is empty, the heuristic is 0 (goal reached for these nuts).
    3. Find the man's name and current location from the state.
    4. Find the locations of all currently usable spanners (on the ground).
    5. Check if the man is currently carrying a usable spanner.
    6. Calculate the total number of usable spanners available (on the ground + carried).
    7. If the number of nuts in N_loose_goal exceeds the total number of usable spanners, the problem is unsolvable from this state, return infinity.
    8. Initialize total heuristic cost to 0.
    9. For each nut 'n' in N_loose_goal:
        a. Get the location of nut 'n', let it be 'l_n'. This location is static and found during initialization.
        b. Estimate the cost to get the man and a usable spanner to 'l_n', and then tighten the nut.
        c. Cost to tighten = 1 (for the 'tighten_nut' action).
        d. Cost to get man and spanner to 'l_n':
            - If the man is currently carrying a usable spanner: This cost is the shortest distance from the man's current location to 'l_n'.
            - If the man is NOT currently carrying a usable spanner: He needs to go to a location 'l_s' with a usable spanner, pick it up (cost 1), and then travel from 'l_s' to 'l_n'. The minimum cost for this part is 1 (pickup) + min_{l_s in usable_spanner_locations} (distance(man_loc, l_s) + distance(l_s, l_n)). If no usable spanners are available on the ground, this part is impossible, leading to infinite cost for this nut.
        e. Add (Cost to get man and spanner to 'l_n') + (Cost to tighten) to the total heuristic cost.
    10. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Identify all locations
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                self.locations.add(parts[1])
                self.locations.add(parts[2])
        # Also need locations from initial state facts like (at obj loc)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 self.locations.add(parts[2])

        # Build adjacency list for the location graph (assuming bidirectional links)
        self.adj = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                if l1 in self.adj and l2 in self.adj: # Ensure locations are in our set
                    self.adj[l1].add(l2)
                    self.adj[l2].add(l1) # Assume bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                (curr, d) = q.popleft()

                for neighbor in self.adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = d + 1
                        q.append((neighbor, d + 1))

        # Identify nut objects and their static locations
        self.nut_locations = {}
        # Nuts are locatable, their initial location is static
        nut_names_in_goal = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                 obj_name, loc_name = parts[1:]
                 # Check if this object is one of the goal nuts
                 if obj_name in nut_names_in_goal:
                     self.nut_locations[obj_name] = loc_name

        # Store goal nuts
        self.goal_nuts = nut_names_in_goal


    def get_distance(self, loc1, loc2):
        """Helper to get shortest path distance, returns infinity if no path."""
        if loc1 not in self.locations or loc2 not in self.locations:
             # This can happen if a location only appears in a state fact, not in init/static
             # Re-computing distances or handling unknown locations would be complex.
             # For this heuristic, assume all relevant locations are in the graph built from init/static.
             # Returning inf means this path is considered impossible.
             return float('inf')
        return self.distances.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose nuts that are goals
        current_loose_nuts = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}
        nuts_to_tighten = self.goal_nuts.intersection(current_loose_nuts)

        # 2. If N_loose_goal is empty, the heuristic is 0.
        if not nuts_to_tighten:
            return 0

        # Find the man's name (assuming there's only one man)
        man_name = None
        # Try finding the object involved in 'carrying'
        for fact in state:
             if match(fact, "carrying", "*", "*"):
                  man_name = get_parts(fact)[1]
                  break
        # If not carrying, try finding the object of type 'man' in initial state
        # (This requires parsing types, which we don't have. Fallback to naming convention).
        if man_name is None:
             # Assume object starting with 'bob' or similar is the man.
             # Or the first object in an 'at' predicate that is not a spanner/nut.
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3:
                       obj_name = parts[1]
                       # Check if it's a spanner or nut (assuming naming convention)
                       if not obj_name.startswith('spanner') and not obj_name.startswith('nut'):
                            man_name = obj_name
                            break
        if man_name is None:
             # print("Warning: Could not identify man object.")
             return float('inf') # Cannot make progress

        # 3. Find the man's current location.
        man_location = None
        for fact in state:
             if match(fact, "at", man_name, "*"):
                  man_location = get_parts(fact)[2]
                  break
        if man_location is None:
             # print(f"Warning: Could not find location for man '{man_name}'.")
             return float('inf') # Cannot make progress


        # 4. Find the locations of all currently usable spanners (on the ground).
        usable_spanner_locations = set()
        usable_spanners_on_ground_count = 0
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, loc_name = get_parts(fact)[1:]
                # Check if this object is a usable spanner on the ground
                is_spanner = obj_name.startswith('spanner') # Assuming naming convention
                is_usable = False
                if is_spanner:
                     for fact_u in state:
                          if match(fact_u, "usable", obj_name):
                               is_usable = True
                               break
                if is_spanner and is_usable:
                    usable_spanner_locations.add(loc_name)
                    usable_spanners_on_ground_count += 1


        # 5. Check if the man is currently carrying a usable spanner.
        man_carrying_usable_spanner = False
        carried_spanner_name = None
        for fact in state:
             if match(fact, "carrying", man_name, "*"):
                  carried_spanner_name = get_parts(fact)[2]
                  # Check if the carried spanner is usable
                  for fact_u in state:
                       if match(fact_u, "usable", carried_spanner_name):
                            man_carrying_usable_spanner = True
                            break
                  break # Found carried spanner info

        # 6. Calculate the total number of usable spanners available (on the ground + carried).
        total_usable_spanners = usable_spanners_on_ground_count + (1 if man_carrying_usable_spanner else 0)

        # 7. If the number of nuts in N_loose_goal exceeds the total number of usable spanners, return infinity.
        if len(nuts_to_tighten) > total_usable_spanners:
             # print(f"Heuristic returning inf: Not enough spanners ({total_usable_spanners}) for nuts ({len(nuts_to_tighten)})")
             return float('inf')


        # 8. Initialize total heuristic cost to 0.
        total_heuristic_cost = 0

        # 9. For each nut 'n' in N_loose_goal:
        for nut in nuts_to_tighten:
             nut_loc = self.nut_locations.get(nut)
             if nut_loc is None:
                  # print(f"Warning: Could not find location for nut '{nut}'.")
                  return float('inf') # Cannot make progress on this nut

             # c. Cost to tighten = 1
             cost_for_nut = 1

             # d. Cost to get man and spanner to 'l_n':
             cost_to_get_man_spanner_to_nut_loc = 0
             if man_carrying_usable_spanner:
                  # Man has spanner, just needs to travel to the nut
                  dist = self.get_distance(man_location, nut_loc)
                  if dist == float('inf'): return float('inf')
                  cost_to_get_man_spanner_to_nut_loc = dist
             else:
                  # Man needs to get a spanner first, then go to the nut
                  min_cost_pickup_and_travel = float('inf')
                  if usable_spanner_locations:
                       for l_s in usable_spanner_locations:
                            dist_man_to_s = self.get_distance(man_location, l_s)
                            dist_s_to_nut = self.get_distance(l_s, nut_loc)
                            if dist_man_to_s != float('inf') and dist_s_to_nut != float('inf'):
                                 min_cost_pickup_and_travel = min(min_cost_pickup_and_travel, dist_man_to_s + 1 + dist_s_to_nut)
                  else:
                       # No usable spanners on the ground and man isn't carrying one
                       # print("Heuristic returning inf: No usable spanners available on ground.")
                       return float('inf') # Cannot get a spanner

                  if min_cost_pickup_and_travel == float('inf'):
                       # print(f"Heuristic returning inf: Cannot reach any usable spanner or nut {nut}.")
                       return float('inf') # Cannot reach a spanner location or the nut location via a spanner

                  cost_to_get_man_spanner_to_nut_loc = min_cost_pickup_and_travel

             # e. Add costs for this nut
             cost_for_nut += cost_to_get_man_spanner_to_nut_loc
             total_heuristic_cost += cost_for_nut

        # 10. Return the total heuristic cost.
        return total_heuristic_cost
