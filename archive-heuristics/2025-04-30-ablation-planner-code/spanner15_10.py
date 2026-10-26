from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # Assuming task object has objects attribute as a list of (name, type) tuples
        self.objects = task.objects

    def __call__(self, node):
        raise NotImplementedError

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_graph(static_facts):
    """Build a bidirectional graph from 'link' predicates."""
    graph = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'link':
            l1, l2 = parts[1], parts[2]
            locations.add(l1)
            locations.add(l2)
            if l1 not in graph: graph[l1] = []
            if l2 not in graph: graph[l2] = []
            graph[l1].append(l2)
            graph[l2].append(l1) # Assume bidirectional links for walk action
    return graph, list(locations)

def bfs(graph, start_node):
    """Compute shortest path distances from start_node to all reachable nodes."""
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
         # Start node might exist but have no links in the graph
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node is still in graph (should be if it was added to queue)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the cost of tightening each nut, picking up necessary spanners,
    and the estimated travel cost for the man to move between spanners and nuts.

    # Assumptions:
    - The man can carry multiple spanners.
    - A spanner becomes unusable after tightening one nut.
    - The location graph defined by 'link' predicates is bidirectional.
    - Nut locations are static.

    # Heuristic Initialization
    - Build the location graph from static 'link' facts.
    - Identify the name of the man object.
    - Identify all nut and spanner objects.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts and their current locations.
    3. Identify all usable spanners and their current locations (either on the ground or carried by the man).
    4. Count the number of loose nuts (`N_loose`), total usable spanners (`N_usable`), and usable spanners currently carried by the man (`N_carried_usable`).
    5. If `N_loose` is 0, the goal is reached, return 0.
    6. If `N_loose > N_usable`, the problem is unsolvable, return infinity.
    7. Calculate the cost for 'tighten_nut' actions: This is simply `N_loose`.
    8. Calculate the cost for 'pickup_spanner' actions: The man needs `N_loose` usable spanners in total. If he is already carrying `N_carried_usable` usable spanners, he needs to pick up `max(0, N_loose - N_carried_usable)` additional usable spanners from the ground. The pickup cost is `max(0, N_loose - N_carried_usable)`.
    9. Estimate the travel cost: The man needs to perform `N_loose` tasks, each involving getting a spanner and going to a nut location.
       - Compute shortest path distances using BFS from the man's current location, all usable spanner locations, and all loose nut locations.
       - Calculate minimum distances between relevant location types:
         - `min_dist_man_to_spanner`: minimum distance from man's location to any usable spanner location.
         - `min_dist_man_to_nut`: minimum distance from man's location to any loose nut location.
         - `min_dist_spanner_to_nut`: minimum distance from any usable spanner location to any loose nut location.
         - `min_dist_nut_to_spanner`: minimum distance from any loose nut location to any usable spanner location.
       - Estimate the total travel cost by modeling a simplified path: Man starts by going to the closest usable spanner, then cycles between spanner and nut locations `N_loose` times. The total travel is estimated as the sum of the initial leg (man to first spanner), `N_loose` legs from spanner to nut, and `N_loose - 1` legs from nut to the next spanner (if `N_loose > 1`).
       - Handle cases where required locations are unreachable (distance is infinity).
    10. The total heuristic value is the sum of the tighten cost, pickup cost, and estimated travel cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and object information."""
        # Assuming task object has static facts and objects list
        self.graph, self.locations = build_graph(task.static)
        self.all_objects = task.objects # List of (name, type) tuples

        # Find the man object name
        self.man_name = None
        for obj_name, obj_type in self.all_objects:
            if obj_type == 'man':
                self.man_name = obj_name
                break
        # Raise error if man is not found as it's essential
        if not self.man_name:
             raise ValueError("Man object not found in task objects.")

        # Find all nut and spanner object names
        self.all_nut_names = [name for name, type in self.all_objects if type == 'nut']
        self.all_spanner_names = [name for name, type in self.all_objects if type == 'spanner']


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify state elements
        man_loc = None
        loose_nuts = set()
        nut_locations = {} # Map nut name to location
        usable_spanners = set() # Names of usable spanners
        usable_spanner_locations = set() # Locations of usable spanners (on ground or man's loc)
        carried_usable_spanners = set() # Names of usable spanners currently carried by the man

        # Find man's location
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break

        if man_loc is None:
             # Man's location must be known in a valid state
             return math.inf

        # Find locations for all nuts (locations are static)
        for nut_name in self.all_nut_names:
             for fact in state:
                 if match(fact, "at", nut_name, "*"):
                     nut_locations[nut_name] = get_parts(fact)[2]
                     break # Assume nut location is unique and static

        # Identify loose nuts
        for nut_name in self.all_nut_names:
             is_loose = False
             for fact in state:
                 if match(fact, "loose", nut_name):
                     is_loose = True
                     break
             if is_loose:
                 loose_nuts.add(nut_name)

        loose_nut_locs = {nut_locations[n] for n in loose_nuts if n in nut_locations}


        # Find usable spanners and their locations
        for spanner_name in self.all_spanner_names:
            is_usable = False
            spanner_current_loc = None

            for fact in state:
                if match(fact, "usable", spanner_name):
                    is_usable = True
                    break # Found usable status

            if is_usable:
                usable_spanners.add(spanner_name)
                # Check if this usable spanner is carried
                is_carried = False
                for carried_fact in state:
                     if match(carried_fact, "carrying", self.man_name, spanner_name):
                         is_carried = True
                         spanner_current_loc = man_loc # Spanner is with the man
                         carried_usable_spanners.add(spanner_name)
                         break # Found carried status
                # If not carried, find its location on the ground
                if not is_carried:
                    for at_fact in state:
                        if match(at_fact, "at", spanner_name, "*"):
                            spanner_current_loc = get_parts(at_fact)[2] # Spanner is on the ground
                            break # Found ground location

            if is_usable and spanner_current_loc:
                usable_spanner_locations.add(spanner_current_loc)


        # 4. Count relevant items
        n_loose = len(loose_nuts)
        n_usable = len(usable_spanners)
        n_carried_usable = len(carried_usable_spanners)

        # 5. Base case: Goal reached
        if n_loose == 0:
            return 0

        # 6. Unsolvable case: Not enough usable spanners
        if n_loose > n_usable:
            return math.inf

        # 7. Tighten cost
        tighten_cost = n_loose

        # 8. Pickup cost
        # Man needs N_loose spanners. If he carries N_carried_usable usable ones,
        # he needs to pick up max(0, N_loose - N_carried_usable) additional usable spanners from the ground.
        pickup_cost = max(0, n_loose - n_carried_usable)

        # 9. Estimate travel cost
        travel_cost = 0

        # Compute distances using BFS
        dist_from_man = bfs(self.graph, man_loc)

        # Collect locations for BFS from spanners and nuts
        spanner_bfs_starts = list(usable_spanner_locations)
        nut_bfs_starts = list(loose_nut_locs)

        # Calculate minimum distances between relevant location types
        min_dist_man_to_spanner = min([dist_from_man.get(loc, math.inf) for loc in usable_spanner_locations], default=math.inf)
        min_dist_man_to_nut = min([dist_from_man.get(loc, math.inf) for loc in loose_nut_locs], default=math.inf)

        min_dist_spanner_to_nut = math.inf
        if usable_spanner_locations and loose_nut_locs:
            dist_from_spanner_locs = {loc: bfs(self.graph, loc) for loc in spanner_bfs_starts}
            for ls in usable_spanner_locations:
                if ls in dist_from_spanner_locs:
                    for ln in loose_nut_locs:
                        min_dist_spanner_to_nut = min(min_dist_spanner_to_nut, dist_from_spanner_locs[ls].get(ln, math.inf))

        min_dist_nut_to_spanner = math.inf
        if loose_nut_locs and usable_spanner_locations:
            dist_from_nut_locs = {loc: bfs(self.graph, loc) for loc in nut_bfs_starts}
            for ln in loose_nut_locs:
                 if ln in dist_from_nut_locs:
                     for ls in usable_spanner_locations:
                         min_dist_nut_to_spanner = min(min_dist_nut_to_spanner, dist_from_nut_locs[ln].get(ls, math.inf))


        # Estimate travel cost based on a simplified sequential path:
        # ManLoc -> First Spanner -> First Nut -> Second Spanner -> Second Nut -> ... -> Last Spanner -> Last Nut
        # This requires:
        # 1. Travel from ManLoc to the closest usable spanner location.
        # 2. N_loose segments of travel from a spanner location to a nut location.
        # 3. N_loose - 1 segments of travel from a nut location to the next spanner location (if N_loose > 1).

        if n_loose > 0:
            # Cost of the first leg: ManLoc to closest usable spanner
            if min_dist_man_to_spanner == math.inf:
                # Cannot reach any usable spanner
                return math.inf
            travel_cost += min_dist_man_to_spanner

            # Cost of N_loose segments from spanner to nut
            if min_dist_spanner_to_nut == math.inf:
                # Cannot reach any nut from any usable spanner
                return math.inf
            travel_cost += n_loose * min_dist_spanner_to_nut

            # Cost of N_loose - 1 segments from nut to spanner (if N_loose > 1)
            if n_loose > 1:
                if min_dist_nut_to_spanner == math.inf:
                    # Cannot reach any usable spanner from any loose nut
                    return math.inf
                travel_cost += (n_loose - 1) * min_dist_nut_to_spanner

        # 10. Total heuristic
        total_heuristic = tighten_cost + pickup_cost + travel_cost

        return total_heuristic
