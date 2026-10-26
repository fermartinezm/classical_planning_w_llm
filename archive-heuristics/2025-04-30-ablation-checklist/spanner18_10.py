from fnmatch import fnmatch
from collections import deque
# Assuming the Heuristic base class is available in a module named heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided
# This is just for standalone testing/syntax checking.
# In the actual planning environment, the real Heuristic class will be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Could not import Heuristic base class. Using a dummy class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic call not implemented")


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
    # Ensure the number of parts matches the number of args, unless args ends with '*'
    if len(parts) != len(args) and (not args or args[-1] != '*'):
        return False
    if args and args[-1] == '*':
         # Match up to the second to last arg, the last arg '*' matches any remaining parts
         if len(parts) < len(args) -1:
             return False
         return all(fnmatch(part, arg) for part, arg in zip(parts[:len(args)-1], args[:-1]))
    else:
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def compute_distances(static_facts):
    """
    Computes shortest path distances between all pairs of locations
    based on the 'link' predicates using BFS.
    Returns a dictionary {(loc1, loc2): distance}.
    Returns float('inf') for unreachable locations.
    """
    locations = set()
    links = set()
    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            l1, l2 = get_parts(fact)[1:]
            locations.add(l1)
            locations.add(l2)
            # Links are bidirectional
            links.add((l1, l2))
            links.add((l2, l1))

    distances = {}
    all_locations = list(locations)

    for start_loc in all_locations:
        queue = deque([(start_loc, 0)])
        visited = {start_loc: 0}
        distances[(start_loc, start_loc)] = 0

        while queue:
            current_loc, dist = queue.popleft()

            # Find neighbors
            neighbors = [l2 for l1, l2 in links if l1 == current_loc]

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    distances[(start_loc, neighbor)] = dist + 1
                    queue.append((neighbor, dist + 1))

    # Ensure all pairs have a distance (infinity if not reachable)
    for l1 in all_locations:
        for l2 in all_locations:
            if (l1, l2) not in distances:
                 distances[(l1, l2)] = float('inf')

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost by greedily simulating the process of tightening
    remaining loose goal nuts. In each step, it finds the minimum cost
    to reach *any* remaining loose goal nut with an available usable spanner,
    adds this cost to the total, and updates the state for the next step
    (man moves to nut location, spanner is consumed).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        object information, and computing location distances.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to find initial locations

        # 1. Identify all nut names
        all_nut_names = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["loose", "tightened"] and len(parts) == 2:
                 all_nut_names.add(parts[1])
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] in ["loose", "tightened"] and len(parts) == 2:
                 all_nut_names.add(parts[1])

        # 2. Identify all spanner names
        all_spanner_names = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == "usable" and len(parts) == 2:
                  all_spanner_names.add(parts[1])
             elif parts[0] == "carrying" and len(parts) == 3:
                  all_spanner_names.add(parts[2]) # The spanner is the second arg

        # 3. Find the man's name
        self.man_name = None
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == "carrying" and len(parts) == 3:
                 self.man_name = parts[1] # Man is the first arg of carrying
                 break
        if self.man_name is None:
             # If no carrying fact, find the object in an 'at' fact that isn't a nut or spanner
             for fact in self.initial_state:
                  parts = get_parts(fact)
                  if parts[0] == "at" and len(parts) == 3:
                       obj_name = parts[1]
                       if obj_name not in all_nut_names and obj_name not in all_spanner_names:
                            self.man_name = obj_name
                            break

        if self.man_name is None:
             # This indicates a malformed problem instance without a clear man object.
             # For a robust planner, this should raise an error.
             # For this heuristic, we'll print a warning and proceed,
             # but it's likely to fail later if man_name is None.
             print("Warning: Could not identify the man object in the initial state.")


        # 4. Store initial locations for nuts and spanners
        # Nut locations are static throughout the problem.
        self.nut_initial_locations = {}
        # Spanner initial locations are needed to know where they start on the ground.
        self.spanner_initial_locations = {}
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                 obj_name = parts[1]
                 location = parts[2]
                 if obj_name in all_nut_names:
                      self.nut_initial_locations[obj_name] = location
                 elif obj_name in all_spanner_names:
                      self.spanner_initial_locations[obj_name] = location

        # 5. Identify goal nuts (those that must be tightened)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # 6. Precompute shortest path distances between all locations
        self.distances = compute_distances(self.static_facts)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        using a greedy simulation.
        """
        state = node.state

        # If man_name was not found during init, we cannot compute the heuristic.
        if self.man_name is None:
             return float('inf') # Indicate unsolvable or invalid state

        # 1. Identify remaining goal nuts that are not tightened
        # These are the nuts in the goal set that do NOT have the (tightened ?) fact in the current state.
        remaining_nuts = {nut for nut in self.goal_nuts if f"(tightened {nut})" not in state}

        if not remaining_nuts:
            return 0 # Goal reached for all nuts

        # 2. Get current man location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man is not at any location? Invalid state.
             return float('inf')

        # 3. Identify currently usable spanners
        usable_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # 4. Identify spanner carried by man (if any). Assuming man carries at most one.
        # The domain definition suggests singular carrying.
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                 carried_spanner = get_parts(fact)[2]
                 break

        # 5. Identify locations of usable spanners on the ground
        available_ground_spanners = {} # {spanner_name: location}
        for spanner in usable_spanners_in_state:
             if spanner != carried_spanner: # Exclude the one being carried
                 # Find where this spanner is located on the ground
                 for fact in state:
                     if match(fact, "at", spanner, "*"):
                         available_ground_spanners[spanner] = get_parts(fact)[2]
                         break # Found location for this spanner

        # Check if there are enough usable spanners for remaining nuts
        # If the number of usable spanners (carried + on ground) is less than the number of remaining nuts,
        # the problem is unsolvable from this state.
        num_usable_available = len(usable_spanners_in_state) # This count includes the carried one if usable
        if num_usable_available < len(remaining_nuts):
             return float('inf')


        # --- Greedy Simulation ---
        h_cost = 0
        current_man_loc = man_loc
        # In the simulation, the man starts carrying a usable spanner if he does so in the current state.
        sim_man_carrying_usable = (carried_spanner is not None and carried_spanner in usable_spanners_in_state)

        # Create a mutable copy of available ground spanners for simulation
        sim_available_ground_spanners = dict(available_ground_spanners)

        # Get locations of remaining nuts (they are static - from initial state)
        remaining_nut_locations = {nut: self.nut_initial_locations.get(nut, None) for nut in remaining_nuts}
        # Check if any remaining nut location is unknown (malformed problem)
        if any(loc is None for loc in remaining_nut_locations.values()):
             print("Error: Location of a goal nut not found in initial state.")
             return float('inf')


        # Simulate tightening nuts one by one
        # In each step, find the cheapest nut to tighten next.
        while remaining_nuts:
            min_step_cost = float('inf')
            best_nut_for_step = None
            spanner_used_in_step = None # Name of the spanner used in this simulation step ("carried" or spanner_name)

            # Calculate cost for each remaining nut using available spanners
            for nut in remaining_nuts:
                nut_location = remaining_nut_locations[nut]

                # Option 1: Use the spanner the man is currently simulated to be carrying (if any)
                if sim_man_carrying_usable:
                    # Cost is walk from current man location to nut location + tighten action
                    dist_to_nut = self.distances.get((current_man_loc, nut_location), float('inf'))
                    if dist_to_nut != float('inf'):
                         cost = dist_to_nut + 1 # walk + tighten
                         if cost < min_step_cost:
                             min_step_cost = cost
                             best_nut_for_step = nut
                             spanner_used_in_step = "carried" # Placeholder

                # Option 2: Pick up an available usable spanner from the ground
                for spanner, s_loc in sim_available_ground_spanners.items():
                    # Cost is walk from current man location to spanner location + pickup + walk from spanner location to nut location + tighten
                    dist_to_spanner = self.distances.get((current_man_loc, s_loc), float('inf'))
                    dist_spanner_to_nut = self.distances.get((s_loc, nut_location), float('inf'))

                    if dist_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                         cost = dist_to_spanner + 1 + dist_spanner_to_nut + 1 # walk1 + pickup + walk2 + tighten
                         if cost < min_step_cost:
                             min_step_cost = cost
                             best_nut_for_step = nut
                             spanner_used_in_step = spanner # Name of the ground spanner

            # If no nut could be tightened in this step (no path to any spanner/nut combo)
            if best_nut_for_step is None:
                 # This implies remaining_nuts is not empty, but min_step_cost is still infinity.
                 # This means no remaining nut is reachable with any available usable spanner.
                 # The problem is likely unsolvable from this state.
                 return float('inf')

            # Add the cost of this step to the total heuristic
            h_cost += min_step_cost

            # Update the simulated state for the next iteration
            remaining_nuts.remove(best_nut_for_step)
            current_man_loc = remaining_nut_locations[best_nut_for_step] # Man is now at the nut location after tightening
            sim_man_carrying_usable = False # The spanner used is consumed

            # If a ground spanner was used, remove it from available ground spanners
            if spanner_used_in_step != "carried":
                 # spanner_used_in_step holds the name of the ground spanner
                 if spanner_used_in_step in sim_available_ground_spanners:
                      del sim_available_ground_spanners[spanner_used_in_step]
                 # else: This spanner was already removed? Should not happen with correct logic.

        # All loose goal nuts have been processed in the simulation
        return h_cost

