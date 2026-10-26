# Helper functions
from collections import deque
from fnmatch import fnmatch

def shortest_paths_from(graph, start):
    """
    Computes shortest path distances from a start node to all reachable nodes
    in an unweighted graph using BFS.

    Args:
        graph: A dictionary where keys are nodes and values are sets of neighbors.
        start: The starting node.

    Returns:
        A dictionary mapping reachable nodes to their distance from the start node.
        Returns {start: 0} if start is in graph but has no neighbors.
        Returns {} if start is not in graph.
    """
    if start not in graph:
        return {}

    distances = {start: 0}
    queue = deque([start])

    while queue:
        current_loc = queue.popleft()
        current_dist = distances[current_loc]

        # Ensure current_loc is a valid key before accessing neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

    return distances

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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming heuristics.heuristic_base.Heuristic is available
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It sums the number of tighten actions, the number of spanner pickup actions needed,
    and an estimate of the movement cost. The movement cost is estimated as the sum
    of shortest path distances from the man's current location to each required
    location (nut locations and locations of needed spanners).

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Spanners are consumed after one use for tightening.
    - The man can carry multiple spanners.
    - Shortest path distances between locations can be computed efficiently using BFS.
    - There is exactly one man object in the domain.
    - Object types (man, nut, spanner, location) can be inferred from predicate usage in the initial state and goals.
    - The location graph is connected for solvable problems.

    # Heuristic Initialization
    - Builds a graph of locations based on `link` predicates from static facts
      to enable shortest path calculations.
    - Identifies the man object name, all nut object names, and all spanner object names
      by examining predicate usage in the initial state and goals.
    - Identifies all location object names.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Identify the man's current location. If unknown or not in the location graph, return infinity.
    2.  Identify all goal nuts that are currently `loose`. Count them (`N_loose`) and record their locations (`nut_locations`). If any loose goal nut's location is unknown, return infinity.
    3.  If `N_loose` is 0, the goal is reached, heuristic is 0.
    4.  Count the number of usable spanners the man is currently carrying (`N_usable_carried`).
    5.  Identify all usable spanners currently on the ground and their locations (`usable_spanners_ground`). Filter out spanners at locations not in the location graph.
    6.  Calculate the number of additional spanners the man needs to pick up from the ground: `needed_pickups = max(0, N_loose - N_usable_carried)`.
    7.  Check if enough usable spanners exist in total (carried + on ground) to tighten all loose goal nuts. If not, return infinity.
    8.  Compute shortest path distances from the man's current location to all other locations using BFS on the location graph.
    9.  Determine the set of required locations the man must visit:
        -   All locations of the loose goal nuts (`nut_locations.values()`).
        -   The locations of the `needed_pickups` usable spanners on the ground that are closest to the man's current location (based on shortest path distance from the man). Filter out spanner locations unreachable from the man.
    10. Check if all required locations (nut locations and selected spanner locations) are reachable from the man's current location. If not, return infinity.
    11. Calculate the movement cost as the sum of the shortest path distances from the man's current location to each location in the set of required locations.
    12. The total heuristic value is the sum of:
        -   `N_loose` (estimated tighten actions).
        -   `needed_pickups` (estimated pickup actions).
        -   Movement cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Object Identification ---
        all_objects = set()
        for fact in initial_state | static_facts | self.goals:
             parts = get_parts(fact)
             if parts: # Ensure fact is not empty after stripping
                 all_objects.update(parts[1:])

        self.man_obj = None
        self.nut_objs = set()
        self.spanner_objs = set()
        all_locations = set()

        # Identify man (object at a location in init and carried in init)
        potential_men_at_loc = {get_parts(f)[1] for f in initial_state if match(f, "at", "*", "*")}
        potential_men_carrying = {get_parts(f)[1] for f in initial_state if match(f, "carrying", "*", "*")}
        men_candidates = potential_men_at_loc.intersection(potential_men_carrying)
        if len(men_candidates) == 1:
             self.man_obj = list(men_candidates)[0]
        # else: Problem might not fit assumptions (e.g., no man, multiple men)

        # Identify nuts (objects loose in init or tightened in goals)
        nuts_loose_init = {get_parts(f)[1] for f in initial_state if match(f, "loose", "*")}
        nuts_tightened_goals = {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}
        self.nut_objs = nuts_loose_init | nuts_tightened_goals

        # Identify spanners (objects usable in init or carried in init)
        spanners_usable_init = {get_parts(f)[1] for f in initial_state if match(f, "usable", "*")}
        spanners_carried_init = {get_parts(f)[2] for f in initial_state if match(f, "carrying", "*", "*")}
        self.spanner_objs = spanners_usable_init | spanners_carried_init

        # Identify locations (objects in 'at' or 'link' that are not man/nut/spanner)
        locatables = {self.man_obj} | self.nut_objs | self.spanner_objs
        # Remove None from locatables if man_obj wasn't found
        locatables.discard(None)

        for obj in all_objects:
             if obj not in locatables:
                 # Check if it appears as a location argument
                 is_location = False
                 for fact in initial_state | static_facts:
                     parts = get_parts(fact)
                     if parts and parts[0] == 'at' and len(parts) == 3 and parts[2] == obj:
                         is_location = True
                         break
                     if parts and parts[0] == 'link' and len(parts) == 3 and (parts[1] == obj or parts[2] == obj):
                         is_location = True
                         break
                 if is_location:
                     all_locations.add(obj)

        # --- Build Location Graph ---
        self.location_graph = {loc: set() for loc in all_locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 in self.location_graph and loc2 in self.location_graph:
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1)

        # Store goal nuts
        self.goal_nuts = nuts_tightened_goals


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_loc = None
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None or man_loc not in self.location_graph:
             # Man's location is unknown or not in the graph (isolated node not handled)
             return float('inf')

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts = {n for n in self.goal_nuts if f"(loose {n})" in state}
        nut_locations = {}
        for nut in loose_goal_nuts:
            found_loc = False
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_locations[nut] = get_parts(fact)[2]
                    found_loc = True
                    break
            if not found_loc:
                 # Loose goal nut location unknown - unsolvable?
                 return float('inf')


        N_loose = len(loose_goal_nuts)

        # 3. If N_loose is 0, goal is reached
        if N_loose == 0:
            return 0

        # 4. Count usable spanners carried
        usable_carried_spanners = {s for s in self.spanner_objs if f"(carrying {self.man_obj} {s})" in state and f"(usable {s})" in state}
        N_usable_carried = len(usable_carried_spanners)

        # 5. Identify usable spanners on ground and their locations
        usable_spanners_ground = {} # {spanner_obj: location}
        for spanner in self.spanner_objs:
             if f"(usable {spanner})" in state and f"(carrying {self.man_obj} {spanner})" not in state:
                 for fact in state:
                     if match(fact, "at", spanner, "*"):
                         spanner_loc = get_parts(fact)[2]
                         if spanner_loc in self.location_graph: # Only consider spanners at known locations
                             usable_spanners_ground[spanner] = spanner_loc
                         break

        # 6. Calculate needed pickups
        needed_pickups = max(0, N_loose - N_usable_carried)

        # 7. Check if enough usable spanners exist in total
        total_usable_spanners = N_usable_carried + len(usable_spanners_ground)
        if N_loose > total_usable_spanners:
             return float('inf') # Not enough spanners in the world

        # 8. Compute shortest paths from man's location
        dist_from_man = shortest_paths_from(self.location_graph, man_loc)

        # 9. Determine required locations
        required_locs_set = set(nut_locations.values())

        if needed_pickups > 0:
            # Get usable spanners on ground sorted by distance from man
            # Filter out spanners at locations unreachable from man (already done in step 5)
            reachable_usable_spanners_ground = {
                s: loc for s, loc in usable_spanners_ground.items() if loc in dist_from_man
            }
            sorted_spanners_ground = sorted(
                reachable_usable_spanners_ground.items(),
                key=lambda item: dist_from_man.get(item[1], float('inf')) # Use inf for unreachable
            )
            # Add locations of the needed_pickups closest spanners
            for i in range(min(needed_pickups, len(sorted_spanners_ground))):
                 spanner_loc = sorted_spanners_ground[i][1]
                 required_locs_set.add(spanner_loc)

        # 10. Check reachability of all required locations
        for loc in required_locs_set:
             if loc not in dist_from_man:
                 # A required location is unreachable
                 return float('inf')

        # 11. Calculate movement cost (Sum of distances from man_loc to each required location)
        movement_cost = sum(dist_from_man[loc] for loc in required_locs_set)

        # 12. Total heuristic
        total_cost = N_loose + needed_pickups + movement_cost

        return total_cost
