from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It counts the necessary tighten and pickup actions and adds an estimate of the
    travel cost involved in visiting the locations of nuts and required spanners.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Each nut requires one tighten action and one usable spanner.
    - A spanner is consumed (becomes unusable) after one tighten action.
    - The man can only carry one spanner at a time (this heuristic simplifies this,
      assuming man can acquire a spanner whenever needed, but counts pickup actions).
    - Travel cost is estimated as the sum of distances from the man's current
      location to each required nut location and each required spanner location,
      divided by 2 to account for path efficiency.

    # Heuristic Initialization
    - Extracts all location objects from the task definition by inspecting facts.
    - Builds a graph of locations based on `link` facts.
    - Computes all-pairs shortest paths between locations using Breadth-First Search (BFS).
    - Identifies the set of nuts that must be tightened in the goal state.
    - Identifies the man object's name from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all nuts that are goals and are currently loose, and find their locations.
    3. Identify all usable spanners (carried or on the ground) and their locations.
    4. Count the total number of loose goal nuts (`N_loose_goals`). If 0, heuristic is 0.
    5. Count the number of usable spanners the man is currently carrying (`N_usable_carried`).
    6. Count the number of usable spanners on the ground (`N_usable_ground`) and list their locations.
    7. If the total number of usable spanners (`N_usable_carried + N_usable_ground`) is less than `N_loose_goals`, the problem is likely unsolvable from this state; return a large value (infinity).
    8. Calculate the number of spanners the man needs to pick up from the ground: `N_pickups_required = max(0, N_loose_goals - N_usable_carried)`.
    9. Identify the locations of usable spanners on the ground.
    10. Select the `N_pickups_required` usable ground spanner locations that are closest to the man's current location.
    11. Calculate the total travel distance needed: Sum of distances from the man's current location to each loose goal nut location, plus the sum of distances from the man's current location to each of the selected `N_pickups_required` spanner locations.
    12. The heuristic value is the sum of:
        - `N_loose_goals` (for the tighten actions).
        - `N_pickups_required` (for the pickup actions).
        - The estimated travel cost (total travel distance calculated in step 11, divided by 2.0).
    13. If any required location is unreachable from the man's current location, return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and identifying goal nuts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state
        self.task_facts = task.facts # Use task.facts to find all possible ground facts

        # 1. Identify all location objects and the man object
        self.locations = set()
        self.man_name = None

        # Infer locations from predicate signatures in all possible facts
        for fact_str in self.task_facts:
             parts = get_parts(fact_str)
             predicate = parts[0]
             if predicate == 'at' and len(parts) == 3:
                 # The 3rd part is a location
                 self.locations.add(parts[2])
             elif predicate == 'link' and len(parts) == 3:
                 # Both 2nd and 3rd parts are locations
                 self.locations.add(parts[1])
                 self.locations.add(parts[2])

        # Try to find the man's name from the initial state
        # Assume the man is the object in an 'at' or 'carrying' fact
        # that is not a spanner or nut mentioned in the initial state or goals.
        initial_locatables = set()
        initial_spanners = set()
        initial_nuts = set()

        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                initial_locatables.add(parts[1])
            elif parts[0] == 'carrying' and len(parts) == 3:
                 initial_spanners.add(parts[2]) # Object being carried is spanner
                 self.man_name = parts[1] # Carrier is the man
            elif parts[0] in ['usable', 'loose', 'tightened'] and len(parts) >= 2:
                 if parts[0] == 'usable':
                      initial_spanners.add(parts[1])
                 elif parts[0] in ['loose', 'tightened']:
                      initial_nuts.add(parts[1])

        # If man wasn't found carrying, find the locatable that isn't a known spanner or nut
        if self.man_name is None:
             for obj in initial_locatables:
                  if obj not in initial_spanners and obj not in initial_nuts:
                       self.man_name = obj
                       break

        if self.man_name is None:
             # This case indicates an issue with problem parsing or definition
             # if a man object exists but wasn't identified.
             # For heuristic computation, we'll return infinity if man_name is needed later.
             print("Warning: Could not identify the man object in __init__. Heuristic may return infinity.")


        # Ensure all locations found are valid keys in adj list
        self.adj = {loc: set() for loc in self.locations}

        # 2. Build location graph from link facts
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 in self.adj and loc2 in self.adj: # Ensure locations were identified
                    self.adj[loc1].add(loc2)
                    self.adj[loc2].add(loc1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.dist = {}
        infinity = float('inf')
        for start_node in self.locations:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.dist[start_node][start_node] = 0 # Distance to self is 0

            while q:
                curr_node, d = q.popleft()

                if curr_node in self.adj: # Ensure curr_node is in adj list
                    for neighbor in self.adj.get(curr_node, []): # Use .get for safety
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.dist[start_node][neighbor] = d + 1
                            q.append((neighbor, d + 1))

            # Fill in unreachable locations with infinity
            for end_node in self.locations:
                 if end_node not in self.dist[start_node]:
                      self.dist[start_node][end_node] = infinity


        # 4. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut_name = get_parts(goal)
                self.goal_nuts.add(nut_name)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        infinity = float('inf')

        # Ensure man_name was identified in __init__
        if self.man_name is None:
             return infinity # Cannot compute heuristic without man's name

        # 1. Identify the man's current location.
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man must be at a location in any valid state.
             return infinity

        # 2. Identify loose goal nuts and their locations.
        loose_goal_nuts_info = {} # {nut_name: nut_loc}
        for nut_name in self.goal_nuts:
            # Check if the nut is loose in the current state
            if f'(loose {nut_name})' in state:
                 # Find the nut's location
                 nut_loc = None
                 for fact in state:
                      if match(fact, "at", nut_name, "*"):
                           nut_loc = get_parts(fact)[2]
                           break
                 # If nut is loose but not at a location, it's an invalid state for tightening.
                 if nut_loc is None:
                      return infinity
                 loose_goal_nuts_info[nut_name] = nut_loc


        # 4. Count loose goal nuts. If 0, goal is reached.
        N_loose_goals = len(loose_goal_nuts_info)
        if N_loose_goals == 0:
            return 0

        # 3. Identify usable spanners and their locations.
        usable_spanners = set()
        for fact in state:
             if match(fact, "usable", "*"):
                  usable_spanners.add(get_parts(fact)[1])

        # 5. Count usable spanners carried by the man.
        N_usable_carried = 0
        for spanner in usable_spanners:
             if f'(carrying {self.man_name} {spanner})' in state:
                  N_usable_carried += 1

        # 6. Count usable spanners on the ground and get their locations.
        usable_ground_spanner_locs = []
        N_usable_ground = 0
        for spanner in usable_spanners:
             if f'(carrying {self.man_name} {spanner})' not in state: # Not carried means on ground
                  spanner_loc = None
                  for fact in state:
                       if match(fact, "at", spanner, "*"):
                            spanner_loc = get_parts(fact)[2]
                            break
                  # If a usable spanner is not carried and not at a location, invalid state.
                  if spanner_loc is None:
                       return infinity
                  usable_ground_spanner_locs.append(spanner_loc)
                  N_usable_ground += 1


        # 7. Check solvability based on spanner count.
        if N_loose_goals > N_usable_carried + N_usable_ground:
             return infinity # Not enough usable spanners exist in the state

        # 8. Calculate the number of spanners the man needs to pick up from the ground.
        N_pickups_required = max(0, N_loose_goals - N_usable_carried)

        # 9 & 10. Identify locations of usable ground spanners needed for pickups.
        # Sort usable ground spanner locations by distance from man_loc
        # Need to handle cases where man_loc or spanner_loc is not in self.dist (shouldn't happen if locations are correctly identified)
        usable_ground_spanner_locs.sort(key=lambda loc: self.dist.get(man_loc, {}).get(loc, infinity))

        # Take the first N_pickups_required locations
        needed_spanner_locs = usable_ground_spanner_locs[:N_pickups_required]

        # Check if any needed location is unreachable from man_loc
        # Check distances for both nut locations and needed spanner locations
        all_needed_locations = list(loose_goal_nuts_info.values()) + needed_spanner_locs
        if any(self.dist.get(man_loc, {}).get(loc, infinity) == infinity for loc in all_needed_locations):
             return infinity


        # 11. Calculate the total travel distance needed.
        # Sum distances from man_loc to each nut location
        travel_to_nuts = sum(self.dist[man_loc][loc] for loc in loose_goal_nuts_info.values())
        # Sum distances from man_loc to each needed spanner location
        travel_to_spanners = sum(self.dist[man_loc][loc] for loc in needed_spanner_locs)

        total_travel_distance = travel_to_nuts + travel_to_spanners

        # 12. Calculate heuristic value.
        # Base actions: 1 tighten per nut, 1 pickup per needed ground spanner
        base_actions_cost = N_loose_goals + N_pickups_required

        # Estimated travel cost: total distance divided by 2.0
        # Use max(0, ...) for travel cost in case total_travel_distance is 0 (e.g., man is at all locations)
        estimated_travel_cost = max(0, total_travel_distance / 2.0)

        # Total heuristic value
        h = base_actions_cost + estimated_travel_cost

        return h
