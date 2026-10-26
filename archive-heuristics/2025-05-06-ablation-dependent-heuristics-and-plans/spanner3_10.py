from fnmatch import fnmatch
from collections import deque
import sys # Import sys for float('inf')

# Assume heuristic_base is available in the heuristics directory
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the planner environment
# This allows the code to be runnable standalone for syntax check, but requires
# the actual base class in the target environment.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """Dummy base class for standalone testing."""
        def __init__(self, task):
            pass
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


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose goal nuts.
    It considers the minimum number of `tighten_nut` actions, the minimum number of
    `pickup_spanner` actions required from the ground, and an estimated travel cost
    for the man to visit all necessary locations (nut locations and spanner pickup locations).

    # Assumptions:
    - The man can carry multiple spanners simultaneously.
    - A spanner becomes unusable after tightening one nut.
    - The man must acquire a usable spanner for each loose goal nut that needs tightening.
    - The travel cost is estimated using a greedy approach to the Traveling Salesperson Problem (TSP)
      visiting required nut and spanner locations. This estimate is not admissible but aims to
      reduce expanded nodes in greedy best-first search.
    - The graph of locations connected by `link` predicates is connected.

    # Heuristic Initialization
    - Identify all goal nuts from the task's goal conditions.
    - Build the location graph based on `link` predicates from static facts.
    - Compute all-pairs shortest paths between locations using Breadth-First Search (BFS).
    - Identify all spanner objects and location objects mentioned in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the man object and his current location. (Assumes one man, identified heuristically).
    2. Identify all goal nuts that are currently `loose` in the state.
    3. If there are no loose goal nuts, the goal is reached, and the heuristic is 0.
    4. Count the number of loose goal nuts (`num_loose_goals`). This is the minimum number of `tighten_nut` actions required.
    5. Identify all spanners that are currently `usable`.
    6. Identify all spanners the man is currently `carrying`.
    7. Count the number of usable spanners the man is currently carrying (`num_carried_usable`).
    8. Calculate the total number of usable spanners available in the current state (carried or on the ground). If this is less than `num_loose_goals`, the problem is likely unsolvable from this state, return infinity.
    9. Calculate the number of additional usable spanners the man needs to pick up from the ground (`needed_from_ground`). This is `max(0, num_loose_goals - num_carried_usable)`. This is the minimum number of `pickup_spanner` actions required from the ground.
    10. Identify all usable spanners currently on the ground and determine their locations.
    11. If `needed_from_ground > 0`, find the `needed_from_ground` usable spanners on the ground that are closest to the man's current location. Identify their locations. These are the spanner pickup locations to visit.
    12. Determine the set of locations the man *must* visit: the location of each loose goal nut, and the locations of the `needed_from_ground` closest usable spanners on the ground.
    13. Estimate the travel cost for the man to start at his current location and visit all required locations. Use a greedy TSP approach: repeatedly move from the current location to the closest unvisited required location until all are visited.
    14. The total heuristic value is the sum of:
        - `num_loose_goals` (for the tighten actions)
        - `needed_from_ground` (for the pickup actions)
        - The estimated travel cost (for the walk actions).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Identify all locations and build location graph
        self.locations = set()
        adj = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}

        # Locations from static links
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.locations.add(l1)
                self.locations.add(l2)
                adj.setdefault(l1, []).append(l2)
                adj.setdefault(l2, []).append(l1) # Links are bidirectional

        # Locations from initial state (where objects are)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 self.locations.add(get_parts(fact)[2])

        # Locations from goals (where objects should end up)
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 self.locations.add(get_parts(goal)[2])


        self.dist = {} # Shortest path distances self.dist[loc1][loc2]

        # Compute all-pairs shortest paths using BFS
        for start_node in self.locations:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                curr_loc, d = q.popleft()
                self.dist[start_node][curr_loc] = d
                if curr_loc in adj:
                    for neighbor in adj[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # Identify all spanners (heuristic: objects appearing in usable or carrying predicates)
        self.all_spanners = set()
        for fact in initial_state:
            if match(fact, "usable", "*"):
                self.all_spanners.add(get_parts(fact)[1])
            elif match(fact, "carrying", "*", "*"):
                 self.all_spanners.add(get_parts(fact)[2])
        for goal in self.goals:
             if match(goal, "usable", "*"):
                 self.all_spanners.add(get_parts(goal)[1])
             elif match(goal, "carrying", "*", "*"):
                  self.all_spanners.add(get_parts(goal)[2])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find man object and his current location (heuristic: assume first object in (at) or (carrying) is the man)
        man_obj = None
        man_loc = None
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj_name = get_parts(fact)[1]
                 # Simple heuristic: Assume the object starting with 'bob' is the man
                 # A proper parser would provide object types.
                 if obj_name.startswith('bob'):
                     man_obj = obj_name
                     man_loc = get_parts(fact)[2]
                     break
             elif match(fact, "carrying", "*", "*"):
                  obj_name = get_parts(fact)[1]
                  if obj_name.startswith('bob'):
                      man_obj = obj_name
                      # Man's location is not directly in (carrying) fact, need to find it from (at man_obj loc)
                      for at_fact in state:
                          if match(at_fact, "at", man_obj, "*"):
                              man_loc = get_parts(at_fact)[2]
                              break
                      # If carrying but no (at) fact for man, this is an inconsistent state
                      if man_loc is None: return float('inf')
                      break

        if man_obj is None or man_loc is None:
             # This state is likely invalid or unreachable if the man is not located
             return float('inf')


        # Find loose goal nuts
        loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}

        if not loose_goal_nuts:
            return 0  # Goal reached

        num_loose_goals = len(loose_goal_nuts)

        # Find usable spanners in the current state
        usable_spanners = {s for s in self.all_spanners if f"(usable {s})" in state}

        # Find spanners carried by the man
        carried_spanners = {s for s in self.all_spanners if f"(carrying {man_obj} {s})" in state}

        # Count usable spanners carried by the man
        num_carried_usable = len(carried_spanners.intersection(usable_spanners))

        # Check if enough usable spanners exist in total for all loose goal nuts
        total_usable_available = len(usable_spanners)
        if total_usable_available < num_loose_goals:
             # Not enough usable spanners in the entire state to tighten all nuts
             return float('inf')

        # Calculate needed spanners from ground
        needed_from_ground = max(0, num_loose_goals - num_carried_usable)

        # Find usable spanners on the ground and their locations
        usable_spanners_on_ground = usable_spanners - carried_spanners # Usable and not carried
        ground_usable_spanner_locs = {}
        for s in usable_spanners_on_ground:
             for fact in state:
                 if match(fact, "at", s, "*"):
                     ground_usable_spanner_locs[s] = get_parts(fact)[2]
                     break # Found location, move to next spanner

        # Identify locations of the needed_from_ground closest usable spanners on the ground
        spanner_pickup_locations = set()
        if needed_from_ground > 0:
            # Sort usable spanners on ground by distance from man_loc
            # Ensure man_loc is in self.dist keys before accessing
            if man_loc not in self.dist:
                 # Man is in an unknown location, likely unreachable state
                 return float('inf')

            sorted_spanners_on_ground = sorted(
                ground_usable_spanner_locs.items(),
                key=lambda item: self.dist[man_loc].get(item[1], float('inf'))
            )
            # Take the locations of the needed_from_ground closest spanners
            num_to_pick = min(needed_from_ground, len(sorted_spanners_on_ground))
            # If num_to_pick < needed_from_ground, it means there aren't enough usable spanners on the ground
            # This case should be caught by the total_usable_available check earlier, but handle defensively.
            if num_to_pick < needed_from_ground:
                 return float('inf') # Cannot get all needed spanners from ground

            for i in range(num_to_pick):
                 spanner_pickup_locations.add(sorted_spanners_on_ground[i][1])

        # Identify locations of loose goal nuts
        nut_locations = set()
        for nut in loose_goal_nuts:
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_locations.add(get_parts(fact)[2])
                    break # Found location, move to next nut

        # Combine required locations the man must visit
        required_locations = nut_locations | spanner_pickup_locations

        # Calculate greedy TSP travel cost
        current_loc = man_loc
        travel_cost = 0
        remaining_required = list(required_locations)

        # If man is already at a required location, count it as visited initially
        if current_loc in remaining_required:
             remaining_required.remove(current_loc)

        while remaining_required:
            # Find the closest required location from the current location
            next_loc = None
            min_dist = float('inf')
            for loc in remaining_required:
                # Ensure distance is computable (all locations should be in self.dist and reachable)
                dist_to_loc = self.dist.get(current_loc, {}).get(loc, float('inf'))
                if dist_to_loc < min_dist:
                    min_dist = dist_to_loc
                    next_loc = loc

            if next_loc is None or min_dist == float('inf'):
                 # Cannot reach remaining required locations from current location
                 return float('inf')

            travel_cost += min_dist
            current_loc = next_loc
            remaining_required.remove(next_loc)

        # Total heuristic
        # h = num_loose_goals (tighten actions) + needed_from_ground (pickup actions) + travel_cost (walk actions)
        h = num_loose_goals + needed_from_ground + travel_cost

        return h

