from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Assume Heuristic base class exists and has __init__(self, task) and __call__(self, node)
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string.
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Handle cases where pattern has fewer args than fact parts (e.g., matching "(at obj loc)" with "at", "*")
    # Or cases where pattern has more args (should not match unless pattern ends with *)
    if len(args) > len(parts) and (not args or args[-1] != '*'):
        return False
    if args and args[-1] == '*':
        # Match prefix
        return len(parts) >= len(args) -1 and all(fnmatch(part, arg) for part, arg in zip(parts[:len(args)-1], args[:-1]))
    else:
        # Match exact number of parts
        return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It considers the number of tighten actions, the number of spanner pickups required,
    and the movement cost for the man to reach the necessary locations (nut locations
    and spanner pickup locations).

    # Assumptions
    - Nuts do not move from their initial locations.
    - Spanners do not move unless picked up by the man.
    - Links between locations are bidirectional (walk action is possible in both directions).
    - The number of usable spanners available (carried or on the ground) must be at least
      the number of loose goal nuts for the problem to be solvable.

    # Heuristic Initialization
    - Identify the man object.
    - Identify all location objects.
    - Build the location graph based on `link` predicates, assuming bidirectionality.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify the set of goal nuts from the task's goal conditions.
    - Store the initial locations of all nuts from the task's initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify which spanners are currently usable (either carried or on the ground).
    3. Identify usable spanners currently on the ground and their locations.
    4. Identify loose nuts and their locations.
    5. Determine the set of loose nuts that are also goal nuts (`GoalLooseNuts`).
    6. If `GoalLooseNuts` is empty, the heuristic is 0 (goal reached for nuts).
    7. Count the total number of loose goal nuts (`k`).
    8. Count the number of usable spanners the man is currently carrying (`c`).
    9. Count the number of usable spanners currently on the ground (`u`).
    10. Check for unsolvability: If `k > c + u`, return a large value (infinity).
    11. Calculate the number of spanners the man needs to pick up from the ground (`p = max(0, k - c)`).
    12. Identify the set of unique locations of the loose goal nuts (`N_locs`).
    13. Get a list of usable spanners on the ground with their locations (`UsableGroundSpannersList`).
    14. If `p > 0`, sort `UsableGroundSpannersList` by the distance of the spanner's location from the man's current location. Select the locations of the first `p` spanners from this sorted list. Let this set be `ClosestS_locs`.
    15. The set of locations the man needs to visit is `RequiredVisitLocs = N_locs`. If `p > 0`, add the locations from `ClosestS_locs` to `RequiredVisitLocs`.
    16. Find the location in `RequiredVisitLocs` that is closest to the man's current location.
    17. The heuristic value is the sum of:
        - The number of tighten actions needed (`k`).
        - The number of pickup actions needed (`p`).
        - The shortest distance from the man's current location to the closest location in `RequiredVisitLocs`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goal_nuts = set()
        # Assuming goal is a conjunction of (tightened ?) facts
        goal_facts_list = []
        if get_parts(task.goals)[0] == 'and':
            goal_facts_list = get_parts(task.goals)[1:]
        else:
            goal_facts_list = [task.goals] # Handle single goal fact

        for goal_fact in goal_facts_list:
            if match(goal_fact, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal_fact)[1])

        self.locations = set()
        self.location_graph = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}
        self.nut_initial_locations = {} # {nut_name: location_name}
        self.object_types = {} # {obj_name: type_name}
        self.man_name = None

        # Extract object types and man name
        for obj_def in task.objects:
             # obj_def is like "bob - man" or "shed location1 gate - location"
             parts = obj_def.split()
             obj_type = parts[-1]
             obj_names = parts[:-2] # Remove '-' and type
             for obj_name in obj_names:
                 self.object_types[obj_name] = obj_type
                 if obj_type == 'man':
                     self.man_name = obj_name
                 elif obj_type == 'location':
                     self.locations.add(obj_name)
                     self.location_graph[obj_name] = [] # Initialize adjacency list

        # Build location graph from static facts
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 in self.locations and loc2 in self.locations:
                    # Assuming bidirectionality for walk action
                    self.location_graph[loc1].append(loc2)
                    self.location_graph[loc2].append(loc1)

        # Store initial nut locations
        for fact in task.init:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)[1:3]
                if obj in self.object_types and self.object_types[obj] == 'nut':
                     self.nut_initial_locations[obj] = loc


        # Compute all-pairs shortest paths
        self.shortest_paths = {} # {(loc1, loc2): distance}
        for start_node in self.locations:
            distances = {node: math.inf for node in self.locations}
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                if current_node not in self.location_graph:
                    continue # Should not happen if all locations are in the graph keys

                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

            for end_node in self.locations:
                self.shortest_paths[(start_node, end_node)] = distances[end_node]

    def get_distance(self, loc1, loc2):
        """Helper to get shortest path distance."""
        # Handle cases where locations might not be in the graph (e.g., initial state parsing issues)
        if loc1 not in self.locations or loc2 not in self.locations:
             return math.inf
        return self.shortest_paths.get((loc1, loc2), math.inf)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None or man_location not in self.locations:
             # Man's location is unknown or invalid, problem state is weird or unsolvable
             return math.inf # Should not happen in valid PDDL states

        # 2. Identify usable spanners (carried or on ground)
        usable_spanners_set = set() # Set of usable spanner names
        for fact in state:
             if match(fact, "usable", "*"):
                 spanner_name = get_parts(fact)[1]
                 if spanner_name in self.object_types and self.object_types[spanner_name] == 'spanner':
                     usable_spanners_set.add(spanner_name)

        # 3. Identify usable spanners on ground and their locations
        usable_ground_spanners_list = [] # List of (spanner_name, location_name)
        spanner_locations = {} # {spanner_name: location_name}

        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj_name, loc_name = get_parts(fact)[1:3]
                 if obj_name in self.object_types and self.object_types[obj_name] == 'spanner':
                     spanner_locations[obj_name] = loc_name

        for spanner_name in usable_spanners_set:
             if spanner_name in spanner_locations: # Check if usable spanner is on the ground
                 usable_ground_spanners_list.append((spanner_name, spanner_locations[spanner_name]))

        # 4. Identify loose nuts and their locations
        loose_nuts = set() # Set of loose nut names
        nut_current_locations = {} # {nut_name: location_name}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'loose' and parts[1] in self.object_types and self.object_types[parts[1]] == 'nut':
                 loose_nuts.add(parts[1])
             elif parts[0] == 'at' and parts[1] in self.object_types and self.object_types[parts[1]] == 'nut':
                 nut_current_locations[parts[1]] = parts[2]


        # 5. Determine loose nuts that are also goal nuts
        goal_loose_nuts = loose_nuts.intersection(self.goal_nuts)

        # 6. If GoalLooseNuts is empty, the heuristic is 0
        if not goal_loose_nuts:
            return 0

        # 7. Count the total number of loose goal nuts (k)
        k = len(goal_loose_nuts)

        # 8. Count usable spanners carried (c)
        carried_spanners = set()
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanners.add(get_parts(fact)[2])
        usable_carried_spanners = carried_spanners.intersection(usable_spanners_set)
        c = len(usable_carried_spanners)

        # 9. Count usable spanners on ground (u)
        u = len(usable_ground_spanners_list)

        # 10. Check for unsolvability (not enough usable spanners in the world)
        if k > c + u:
            return math.inf

        # 11. Calculate spanners to pick up (p)
        p = max(0, k - c)

        # 12. Identify locations of loose goal nuts (N_locs)
        n_locs = set()
        for nut in goal_loose_nuts:
             loc = nut_current_locations.get(nut, self.nut_initial_locations.get(nut))
             # Only consider locations that exist and are reachable from the man
             if loc in self.locations and self.get_distance(man_location, loc) != math.inf:
                 n_locs.add(loc)
             else:
                 # A goal nut is at an unreachable location, problem is unsolvable
                 return math.inf


        # 13. Usable ground spanners list is already available

        # 14. Select p locations from usable_ground_spanners_list closest to the man
        # Sort usable ground spanners by distance from man
        # Filter out spanners at unreachable locations first
        reachable_usable_ground_spanners = [(s, l) for s, l in usable_ground_spanners_list if self.get_distance(man_location, l) != math.inf]

        if len(reachable_usable_ground_spanners) < p:
             # Need p spanners from ground, but fewer than p are reachable
             return math.inf # Unsolvable

        reachable_usable_ground_spanners.sort(key=lambda item: self.get_distance(man_location, item[1]))

        # Get the locations of the first p usable ground spanners
        closest_s_locs = set()
        spanners_accounted = 0
        for spanner_name, loc in reachable_usable_ground_spanners:
             if spanners_accounted < p:
                 closest_s_locs.add(loc)
                 spanners_accounted += 1
             else:
                 break # Found enough spanners

        # 15. Required visit locations
        required_visit_locs = n_locs.union(closest_s_locs)

        # 16. Find closest location in RequiredVisitLocs from man_location
        closest_required_dist = math.inf
        if required_visit_locs:
            # All locations in required_visit_locs are guaranteed reachable from man_location
            closest_required_dist = min(self.get_distance(man_location, loc) for loc in required_visit_locs)

        # 17. Heuristic calculation
        # Base cost: k tighten actions + p pickup actions
        base_cost = k + p

        # Movement cost: cost to reach the first required location
        # If required_visit_locs was empty, k must have been 0, handled earlier.
        # If required_visit_locs is not empty, closest_required_dist is finite.
        movement_cost = closest_required_dist

        heuristic_value = base_cost + movement_cost

        return heuristic_value
