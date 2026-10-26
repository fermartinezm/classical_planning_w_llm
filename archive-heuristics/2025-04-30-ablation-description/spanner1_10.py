from fnmatch import fnmatch
from collections import deque
import math
from heuristics.heuristic_base import Heuristic

# Helper functions used by the heuristic
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` are allowed in args.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_paths(locations, links):
    """
    Computes shortest path distances between all pairs of locations
    using BFS on the link graph.
    """
    graph = {loc: set() for loc in locations}
    for l1, l2 in links:
        graph[l1].add(l2)
        graph[l2].add(l1) # Links are bidirectional

    distances = {}
    for start_node in locations:
        distances[start_node] = {loc: math.inf for loc in locations}
        distances[start_node][start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[start_node][current_node]

            for neighbor in graph.get(current_node, []):
                if distances[start_node][neighbor] == math.inf:
                    distances[start_node][neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all loose goal nuts.
    It sums the number of loose goal nuts (representing the tighten actions),
    the estimated movement cost to reach the first nut, and the estimated
    cost to acquire enough usable spanners for all nuts.

    # Assumptions
    - The man can only carry one spanner at a time.
    - Tightening a nut makes the spanner unusable.
    - Links between locations are bidirectional.
    - The cost of any action (walk, pickup, tighten) is 1.

    # Heuristic Initialization
    - Extracts the goal conditions to identify goal nuts.
    - Extracts static facts (`link` predicates) to build the location graph.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the man object from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are goals and are currently loose in the state. Let this set be U.
    2. If U is empty, the heuristic is 0 (goal reached).
    3. Find the man's current location. If unknown, return infinity.
    4. Find all usable spanners currently on the ground and their locations.
    5. Check if the man is currently carrying a usable spanner.
    6. Calculate the number of usable spanners needed from the ground: |U| - (1 if man is carrying a usable spanner else 0). Ensure this count is not negative.
    7. If the number of needed spanners from the ground exceeds the number of usable spanners available on the ground (and reachable), the problem is unsolvable from this state; return infinity.
    8. Calculate the cost to acquire the needed spanners: Find the `num_spanners_needed_from_ground` usable spanners on the ground that are closest to the man's current location (and reachable). Sum their distances from the man's location, adding 1 for each pickup action. If not enough reachable spanners, return infinity.
    9. Calculate the movement cost to reach the first nut: Find the loose goal nut closest to the man's current location (and reachable). The cost is the shortest distance to that nut's location. If no loose goal nuts are reachable (and there are loose goal nuts), return infinity.
    10. The total heuristic value is the sum of:
        - The number of loose goal nuts (|U|), representing the tighten actions.
        - The movement cost to reach the first nut.
        - The cost to acquire the needed spanners from the ground.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        # The base class Heuristic is expected to provide task.goals, task.static, task.initial_state
        super().__init__(task)

        # 1. Build location graph and compute distances
        locations = set()
        links = []
        # Find all locations and links from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                links.append((l1, l2))
                locations.add(l1)
                locations.add(l2)

        # Locations can also appear in initial state facts like (at obj loc)
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 locations.add(loc)

        # Add locations from goals if they appear in 'at' predicates
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 locations.add(loc)

        # Ensure all locations mentioned in links are in the set
        for l1, l2 in links:
            locations.add(l1)
            locations.add(l2)

        self.distances = bfs_shortest_paths(list(locations), links)

        # 2. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # 3. Identify the man object (assume there's only one man)
        self.man = None
        # Look for an object that is 'at' a location and is mentioned in a 'carrying' predicate in the initial state
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                 obj = parts[1]
                 # Check if this object appears in any 'carrying' predicate in initial state
                 is_man_candidate = False
                 for other_fact in self.initial_state:
                     other_parts = get_parts(other_fact)
                     if other_parts and other_parts[0] == 'carrying' and len(other_parts) == 3 and other_parts[1] == obj:
                         is_man_candidate = True
                         break
                 if is_man_candidate:
                     self.man = obj
                     break # Found the man

        # Fallback: If man not found via 'carrying', just find the first object in an (at ...) predicate
        if self.man is None:
             for fact in self.initial_state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'at' and len(parts) == 3:
                     self.man = parts[1]
                     break
             # This fallback is weak if the first object is not the man.
             # A robust parser would provide object types.

        # If man is still None, the initial state is malformed or doesn't fit assumptions
        if self.man is None:
             # In a real system, you might log a warning or raise an error.
             # The heuristic will likely return infinity later if man_loc is not found
             pass


    def __call__(self, node):
        """Estimate the minimum cost to tighten all remaining loose goal nuts."""
        state = node.state

        # 1. Identify loose goal nuts and their locations
        loose_goal_nuts = {} # {nut_name: location}
        for nut in self.goal_nuts:
            # Check if the nut is currently loose
            if f"(loose {nut})" in state:
                 # Find the nut's location
                 nut_loc = None
                 for fact in state:
                     if match(fact, "at", nut, "*"):
                         nut_loc = get_parts(fact)[2]
                         break
                 if nut_loc:
                     loose_goal_nuts[nut] = nut_loc
                 # else: nut is loose but not at a location? Problematic state. Assume it must be at a location.
            # else: nut is already tightened or not a goal nut, ignore.

        # 2. If all goal nuts are tightened, heuristic is 0.
        if not loose_goal_nuts:
            return 0

        # 3. Find man's current location
        man_loc = None
        if self.man: # Ensure man object was identified in __init__
            for fact in state:
                if match(fact, "at", self.man, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None:
             # Man is not at any location or man object wasn't identified
             return math.inf

        # 4. Find usable spanners on the ground and their locations
        usable_spanners_on_ground = {} # {spanner_name: location}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # Check if this usable spanner is on the ground (not carried)
                is_carried = False
                if self.man: # Ensure man object was identified
                    for carry_fact in state:
                        if match(carry_fact, "carrying", self.man, spanner):
                            is_carried = True
                            break
                if not is_carried:
                    # Find its location
                    spanner_loc = None
                    for at_fact in state:
                        if match(at_fact, "at", spanner, "*"):
                            spanner_loc = get_parts(at_fact)[2]
                            break
                    if spanner_loc:
                        usable_spanners_on_ground[spanner] = spanner_loc
                    # else: usable spanner not carried but not at a location? Problematic.

        # 5. Check if man is carrying a usable spanner
        man_carrying_usable = False
        carried_spanner = None
        if self.man: # Ensure man object was identified
            for fact in state:
                if match(fact, "carrying", self.man, "*"):
                    carried_spanner = get_parts(fact)[2]
                    if f"(usable {carried_spanner})" in state:
                        man_carrying_usable = True
                    break # Assuming man carries at most one spanner

        # 6. Calculate number of spanners needed from the ground
        num_nuts_to_tighten = len(loose_goal_nuts)
        num_spanners_needed_from_ground = num_nuts_to_tighten - (1 if man_carrying_usable else 0)
        num_spanners_needed_from_ground = max(0, num_spanners_needed_from_ground) # Cannot need negative spanners

        # 7. & 8. Calculate cost to acquire needed spanners from the ground
        spanner_acquisition_cost = 0
        if num_spanners_needed_from_ground > 0:
            # Find distances from man_loc to all usable spanners on ground
            spanner_distances = [] # List of (distance, spanner_name, location)
            for spanner, s_loc in usable_spanners_on_ground.items():
                 # Ensure man_loc and spanner_loc are valid keys in distances
                 if man_loc in self.distances and s_loc in self.distances[man_loc]:
                    dist = self.distances[man_loc][s_loc]
                    if dist != math.inf: # Only consider reachable spanners
                        spanner_distances.append((dist, spanner, s_loc))

            # If we need spanners but not enough are reachable, it's unsolvable
            if num_spanners_needed_from_ground > len(spanner_distances):
                 return math.inf

            # Sort by distance and take the closest ones
            spanner_distances.sort()
            closest_needed_spanners = spanner_distances[:num_spanners_needed_from_ground]

            # Sum acquisition costs (distance + 1 for pickup)
            spanner_acquisition_cost = sum(dist + 1 for dist, spanner, s_loc in closest_needed_spanners)

        # 9. Calculate movement cost to reach the first nut
        # Find the closest loose goal nut location
        closest_nut_dist = math.inf
        for nut_loc in loose_goal_nuts.values():
             # Ensure man_loc and nut_loc are valid keys in distances
             if man_loc in self.distances and nut_loc in self.distances[man_loc]:
                dist = self.distances[man_loc][nut_loc]
                closest_nut_dist = min(closest_nut_dist, dist)

        # If there are loose goal nuts but none are reachable, problem is unsolvable
        if num_nuts_to_tighten > 0 and closest_nut_dist == math.inf:
             return math.inf

        movement_cost_to_first_nut = closest_nut_dist if num_nuts_to_tighten > 0 else 0


        # 10. Calculate total heuristic
        # Base cost: 1 action per loose goal nut (tighten)
        base_tighten_cost = num_nuts_to_tighten

        # Total heuristic is sum of base actions, cost to reach first nut, and cost to get spanners
        total_cost = base_tighten_cost + movement_cost_to_first_nut + spanner_acquisition_cost

        return total_cost
