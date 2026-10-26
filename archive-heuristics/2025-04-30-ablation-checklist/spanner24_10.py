from fnmatch import fnmatch
# Assuming Heuristic base class is available in the environment like this:
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class is defined as provided in the problem description context
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         self.goals = task.goals
#         self.initial_state = task.initial_state
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost by summing the minimum costs of individual
    nut-tightening tasks, where the cost for a task is the cost
    to get the man and a required usable spanner to the nut's location
    and perform the tightening. It uses a greedy assignment to pair
    nuts and spanners, preventing reuse in the calculation.

    # Heuristic Initialization
    - Identifies all locations mentioned in the problem.
    - Computes all-pairs shortest paths between locations based on 'link' facts.
    - Identifies the man object and the goal nuts from the task definition.

    # Step-By-Step Thinking for Computing Heuristic (__call__)
    1. Identify the man's current location in the given state.
    2. Identify all loose nuts in the state that are also goal conditions, and find their static locations.
    3. Identify all usable spanners in the state and determine their current locations (either on the ground or carried by the man).
    4. If the number of usable spanners is less than the number of loose goal nuts, the problem is unsolvable from this state (return infinity).
    5. Create a list of potential (cost, nut, spanner) tasks:
       - For each loose goal nut N at its location L_N:
       - For each usable spanner S at its location L_S (on ground):
         Calculate the estimated cost to use this spanner for this nut:
         Cost = (walk from man_loc to L_S) + (pickup S) + (walk from L_S to L_N) + (tighten N)
         Cost = dist(man_loc, L_S) + 1 + dist(L_S, L_N) + 1
         Add (cost, N, S) to the list of potential tasks.
       - For each usable spanner S carried by the man:
         Calculate the estimated cost to use this spanner for this nut:
         Cost = (walk from man_loc to L_N) + (tighten N)
         Cost = dist(man_loc, L_N) + 1
         Add (cost, N, S) to the list of potential tasks.
    6. Sort the list of potential tasks by cost in ascending order.
    7. Greedily select tasks from the sorted list:
       - Initialize total_cost = 0, sets of assigned_nuts and assigned_spanners.
       - Iterate through the sorted tasks.
       - If the nut and spanner in the current task have not been assigned yet:
         - If the task cost is infinity, and not all nuts are assigned, the goal is unreachable (return infinity).
         - Add the task's cost to the total heuristic value.
         - Mark the nut and spanner as assigned.
         - If all loose goal nuts have been assigned, stop.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and identifying key objects."""
        super().__init__(task)

        # 1. Identify all locations
        self.all_locations = set()
        # From initial state 'at' facts
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.all_locations.add(loc)
        # From static 'link' facts
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
        self.all_locations = list(self.all_locations) # Convert to list for consistent ordering if needed, set is fine for iteration

        # 2. Build adjacency list for locations
        self.adj_list = {loc: set() for loc in self.all_locations}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.adj_list[loc1].add(loc2)
                self.adj_list[loc2].add(loc1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_locations:
            self.distances[start_node] = {}
            queue = [(start_node, 0)]
            visited = {start_node}
            while queue:
                current_loc, dist = queue.pop(0)
                self.distances[start_node][current_loc] = dist

                for neighbor in self.adj_list.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Handle unreachable locations by setting distance to infinity
        for l1 in self.all_locations:
             for l2 in self.all_locations:
                 if l2 not in self.distances[l1]:
                     self.distances[l1][l2] = float('inf')


        # 4. Identify goal nuts
        self.goal_nuts = set()
        # Goal is a conjunction, iterate through individual goals
        # Assuming goals are always in the form (tightened ?n)
        for goal_fact in self.goals:
             if match(goal_fact, "tightened", "*"):
                 _, nut = get_parts(goal_fact)
                 self.goal_nuts.add(nut)

        # 5. Identify the man object (assuming only one man)
        # Find the object that is locatable but not a spanner or nut based on initial state predicates
        all_spanners_and_nuts = set()
        # Find all spanners mentioned in initial state (usable or at)
        for fact in self.initial_state:
             if match(fact, "usable", "*"):
                 _, spanner = get_parts(fact)
                 all_spanners_and_nuts.add(spanner)
             # Objects at locations might be spanners, nuts, or the man
             # We'll identify spanners/nuts by other predicates (usable, loose, tightened)

        # Find all nuts mentioned in initial state (loose or at) or goals (tightened)
        for fact in self.initial_state:
             if match(fact, "loose", "*"):
                 _, nut = get_parts(fact)
                 all_spanners_and_nuts.add(nut)

        for goal_fact in self.goals:
             if match(goal_fact, "tightened", "*"):
                 _, nut = get_parts(goal_fact)
                 all_spanners_and_nuts.add(nut)


        # Find the man object: the locatable object in initial state that isn't a spanner or nut
        self.man_obj = None
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 # Check if this object is NOT in the set of identified spanners/nuts AND is NOT a location itself
                 if obj not in all_spanners_and_nuts and obj not in self.all_locations:
                     self.man_obj = obj
                     break # Assuming only one man

        # Fallback: If man not found by exclusion, try finding the object in initial state 'at'
        # that is the first argument of a 'carrying' predicate in the initial state.
        if not self.man_obj:
             for fact in self.initial_state:
                  if match(fact, "at", "*", "*"):
                      _, obj, loc = get_parts(fact)
                      if obj in self.all_locations: continue # Skip locations
                      # Check if this object is ever mentioned as carrying something in initial state
                      is_carrier = False
                      for any_fact in self.initial_state:
                           if match(any_fact, "carrying", obj, "*"):
                                is_carrier = True
                                break
                      if is_carrier:
                           self.man_obj = obj
                           break

        # Final fallback: just take the first object in an 'at' predicate that isn't a location
        # This is the least safe but might be needed for minimal instances.
        if not self.man_obj:
             for fact in self.initial_state:
                  if match(fact, "at", "*", "*"):
                      _, obj, loc = get_parts(fact)
                      if obj in self.all_locations: continue
                      self.man_obj = obj
                      break

        if not self.man_obj:
             print("Warning: Could not identify man object.")
             # Heuristic might fail or return 0 incorrectly if man isn't found


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_loc = None
        if self.man_obj: # Proceed only if man object was identified
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    _, _, man_loc = get_parts(fact)
                    break

        if not man_loc:
             # Man not found in state (e.g., man_obj wasn't identified, or state is malformed)
             # Return infinity as we cannot proceed
             return float('inf')


        # 2. Identify loose nuts that are goal conditions and their locations
        loose_goal_nuts_in_state = {} # {nut_name: location}
        for fact in state:
            if match(fact, "loose", "*"):
                _, nut = get_parts(fact)
                if nut in self.goal_nuts:
                    # Find nut's location (nuts are static)
                    nut_loc = None
                    # Search initial state for nut location (assuming nuts don't move)
                    for init_fact in self.initial_state:
                         if match(init_fact, "at", nut, "*"):
                             _, _, nut_loc = get_parts(init_fact)
                             break
                    if nut_loc:
                         loose_goal_nuts_in_state[nut] = nut_loc
                    else:
                         # Nut location not found? Problematic state or task definition.
                         return float('inf') # Or handle as unsolvable


        # 3. Identify usable spanners and their locations
        usable_spanners_at_locs = {} # {spanner_name: location}
        usable_spanners_carried = set() # {spanner_name}

        for fact in state:
            if match(fact, "usable", "*"):
                _, spanner = get_parts(fact)
                # Check if the man is carrying this spanner in the current state
                is_carried = False
                if self.man_obj: # Check only if man object was identified
                    for carry_fact in state:
                         if match(carry_fact, "carrying", self.man_obj, spanner):
                              is_carried = True
                              break
                if is_carried:
                     usable_spanners_carried.add(spanner)
                else:
                     # Spanner is usable but not carried, find its location in the current state
                     spanner_loc = None
                     for at_fact in state:
                          if match(at_fact, "at", spanner, "*"):
                               _, _, spanner_loc = get_parts(at_fact)
                               break
                     if spanner_loc:
                          usable_spanners_at_locs[spanner] = spanner_loc
                     else:
                          # Usable spanner not at a location and not carried? Problematic state.
                          # Assuming 'at' or 'carrying' covers all locatable objects.
                          # If we can't find its location, it's unreachable for pickup.
                          return float('inf')


        # 4. Check if enough usable spanners exist for the remaining nuts
        num_loose_goal_nuts = len(loose_goal_nuts_in_state)
        num_usable_spanners = len(usable_spanners_at_locs) + len(usable_spanners_carried)

        if num_loose_goal_nuts == 0:
            return 0 # Goal reached for nuts

        if num_usable_spanners < num_loose_goal_nuts:
            return float('inf') # Not enough spanners to tighten all required nuts

        # 5. Create potential tasks (cost, nut, spanner)
        potential_tasks = []

        # Spanners at locations
        for nut, nut_loc in loose_goal_nuts_in_state.items():
            for spanner, spanner_loc in usable_spanners_at_locs.items():
                # Cost: walk man to spanner, pickup, walk man to nut, tighten
                # Get distances, defaulting to infinity if location is not in graph (shouldn't happen if locations are identified correctly)
                dist_man_to_spanner = self.distances.get(man_loc, {}).get(spanner_loc, float('inf'))
                dist_spanner_to_nut = self.distances.get(spanner_loc, {}).get(nut_loc, float('inf'))

                if dist_man_to_spanner == float('inf') or dist_spanner_to_nut == float('inf'):
                     cost = float('inf')
                else:
                     cost = dist_man_to_spanner + 1 + dist_spanner_to_nut + 1
                potential_tasks.append((cost, nut, spanner))

        # Spanners carried by man
        for nut, nut_loc in loose_goal_nuts_in_state.items():
             for spanner in usable_spanners_carried:
                 # Cost: walk man to nut, tighten (pickup cost is 0 as already carried)
                 dist_man_to_nut = self.distances.get(man_loc, {}).get(nut_loc, float('inf'))

                 if dist_man_to_nut == float('inf'):
                      cost = float('inf')
                 else:
                      cost = dist_man_to_nut + 1
                 potential_tasks.append((cost, nut, spanner))


        # 6. Sort tasks by cost
        potential_tasks.sort()

        # 7. Greedily select tasks
        total_cost = 0
        nuts_assigned = set()
        spanners_assigned = set()

        for cost, nut, spanner in potential_tasks:
            # If the current cheapest task is impossible (infinite cost)
            # and we still have nuts to assign, the goal is unreachable.
            if cost == float('inf') and len(nuts_assigned) < num_loose_goal_nuts:
                 return float('inf')

            # If the nut and spanner are not already used in a greedy assignment
            if nut not in nuts_assigned and spanner not in spanners_assigned:
                total_cost += cost
                nuts_assigned.add(nut)
                spanners_assigned.add(spanner)

                # If we have assigned a spanner to every loose goal nut, we are done
                if len(nuts_assigned) == num_loose_goal_nuts:
                    break

        # If we finished the loop but didn't assign all nuts, it means there were
        # not enough *reachably located* usable spanners for the loose goal nuts
        # that could be uniquely assigned by the greedy approach.
        # The check inside the loop handles the case where the *cheapest* remaining
        # task is infinite. This final check handles the case where there might
        # be some finite cost tasks, but not enough unique spanners/nuts pairings
        # to cover all required nuts.
        if len(nuts_assigned) < num_loose_goal_nuts:
             return float('inf')


        # 8. Return total heuristic value
        return total_cost

