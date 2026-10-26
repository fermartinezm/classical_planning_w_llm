from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start):
    """Computes shortest path distances from start to all other nodes in a graph."""
    distances = {node: float('inf') for node in graph}
    if start not in graph:
         # Start node is not in the graph of known locations
         return distances # All distances remain infinity

    distances[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose goal nuts.
    It sums the estimated minimum cost for each loose goal nut independently. The cost
    for a single nut is estimated as the minimum actions to get the man to the nut's
    location with a usable spanner, plus the tighten action itself.

    # Assumptions
    - Only nuts specified in the goal need to be tightened.
    - Nuts stay at their initial location.
    - Spanners become 'unusable' after one 'tighten_nut' action.
    - The man can carry multiple spanners simultaneously (inferred from action definition).
    - The man must be at the same location as the nut and carrying a usable spanner to tighten it.
    - Travel between linked locations costs 1 action. Pickup and Tighten actions cost 1.

    # Heuristic Initialization
    - Identify the man, all spanners, and all nuts from the initial state.
    - Build a graph of locations based on 'link' predicates from static facts and locations mentioned in initial/goal states.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the set of goal nuts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Determine which spanners the man is carrying and how many of them are usable.
    3. Identify the locations of all usable spanners currently on the ground.
    4. Identify all goal nuts that are still 'loose' and their locations.
    5. If there are no loose goal nuts, the heuristic is 0 (goal state).
    6. Check if the total number of usable spanners available (carried + on ground) is less than the number of loose goal nuts. If so, the problem is unsolvable from this state, return infinity.
    7. For each loose goal nut:
        a. Get the location of the nut.
        b. Calculate the minimum actions required to get the man to the nut's location *with* a usable spanner, ready to tighten. This is the minimum of two options:
           - Option A: Use one of the usable spanners the man is currently carrying (if any are available). The cost is the shortest distance from the man's current location to the nut's location (walk actions).
           - Option B: Pick up a usable spanner from the ground and go to the nut's location. The minimum cost is found by considering all usable spanners on the ground: minimum over all such spanners S at location L_S of (distance from man's current location to L_S + 1 (pickup) + distance from L_S to the nut's location).
           - The cost to reach the nut with a spanner is the minimum of Option A (if applicable) and Option B (if applicable).
        c. If no usable spanner can be acquired and brought to the nut location, this nut cannot be tightened, return infinity.
        d. Add 1 for the 'tighten_nut' action itself.
        e. Sum this cost for all loose goal nuts.
    8. Return the total summed cost.

    Note: This heuristic sums the costs for each nut independently, which may overestimate the true cost as travel paths and spanner pickups can be shared. However, it captures the essential requirements for each nut and provides a reasonable estimate for greedy search.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and identifying objects."""
        self.goals = task.goals
        self.static_facts = task.static

        # Identify all objects and their potential types from initial state and goals
        self.man_name = None
        self.all_spanners = set()
        self.all_nuts = set()
        all_locations_mentioned = set()

        # Parse initial state to find objects and locations
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                all_locations_mentioned.add(loc)
                if obj.startswith("spanner"):
                    self.all_spanners.add(obj)
                elif obj.startswith("nut"):
                    self.all_nuts.add(obj)
                # Assume the object that is 'at' a location and is not a spanner/nut is the man
                # This is a weak assumption, but works for the examples.
                # A better way requires parsing PDDL object types.
                # If man starts carrying nothing, this finds him.
                if not obj.startswith("spanner") and not obj.startswith("nut"):
                     self.man_name = obj # Found the man (assuming only one)
            elif parts[0] == "carrying" and len(parts) == 3:
                 m, s = parts[1], parts[2]
                 self.man_name = m # Found the man
                 if s.startswith("spanner"):
                      self.all_spanners.add(s)
            # Add other predicates if they mention objects/locations not covered

        # Parse goal state to find goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "tightened" and len(parts) == 2:
                  self.goal_nuts.add(parts[1])
             # Assuming goals are only about tightened nuts for this domain

        # Build location graph from static link facts and all mentioned locations
        self.location_graph = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1)
                all_locations_mentioned.add(l1)
                all_locations_mentioned.add(l2)

        # Ensure all mentioned locations are keys in the graph, even if isolated
        for loc in all_locations_mentioned:
             self.location_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.location_graph:
            self.distances[start_loc] = bfs(self.location_graph, start_loc)

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, handling unreachable locations."""
        # If either location wasn't in the graph built during init, they are unreachable from known locations.
        if loc1 not in self.distances or loc2 not in self.distances:
             return float('inf')
        # BFS result already contains float('inf') for unreachable nodes within the graph.
        return self.distances[loc1].get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify the man's current location.
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break

        if man_loc is None:
             # Man's location not found? State is likely malformed.
             return float('inf')

        # 2. Determine which spanners the man is carrying and how many are usable.
        carried_spanners = set()
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanners.add(get_parts(fact)[2])

        usable_carried_spanners_count = 0
        for s in carried_spanners:
             if f"(usable {s})" in state:
                  usable_carried_spanners_count += 1

        # 3. Identify the locations of all usable spanners currently on the ground.
        usable_spanners_on_ground_locs = set()
        usable_spanners_on_ground_count = 0
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                # Check if obj is a spanner and is usable
                if obj in self.all_spanners and f"(usable {obj})" in state:
                     usable_spanners_on_ground_locs.add(loc)
                     usable_spanners_on_ground_count += 1

        # 4. Identify all goal nuts that are still 'loose' and their locations.
        loose_goal_nuts = {} # {nut_name: location}
        for nut_name in self.goal_nuts:
             # Check if the goal nut is currently loose
             if f"(loose {nut_name})" in state:
                  # Find the location of this nut
                  nut_loc = None
                  for fact in state:
                       if match(fact, "at", nut_name, "*"):
                            nut_loc = get_parts(fact)[2]
                            break
                  if nut_loc is not None:
                       loose_goal_nuts[nut_name] = nut_loc
                  # else: nut location not found? Should not happen in valid states.

        # 5. If there are no loose goal nuts, the heuristic is 0 (goal state).
        if not loose_goal_nuts:
            return 0

        # 6. Check solvability based on spanners
        # Total usable spanners available = usable carried + usable on ground
        total_usable_spanners_available = usable_carried_spanners_count + usable_spanners_on_ground_count

        if len(loose_goal_nuts) > total_usable_spanners_available:
             # Not enough usable spanners exist anywhere to tighten all loose goal nuts
             return float('inf')

        # 7. Sum the estimated cost for each loose goal nut independently.
        total_h = 0

        for nut_name, nut_loc in loose_goal_nuts.items():
            # Calculate the minimum actions required to get the man to the nut's location *with* a usable spanner.
            cost_reach_nut_with_spanner = float('inf')

            # Option A: Use one of the usable spanners currently carried (if available)
            if usable_carried_spanners_count > 0:
                 # Cost is just walking from man's current location to the nut location
                 cost_option_carried = self.get_distance(man_loc, nut_loc)
                 cost_reach_nut_with_spanner = min(cost_reach_nut_with_spanner, cost_option_carried)

            # Option B: Pick up a usable spanner from the ground and go to the nut location
            if usable_spanners_on_ground_locs:
                 # Minimum cost path: ManLoc -> L_S (pickup) -> NutLoc
                 cost_pickup_then_walk_to_nut = float('inf')
                 for l_s in usable_spanners_on_ground_locs:
                      dist_man_to_ls = self.get_distance(man_loc, l_s)
                      dist_ls_to_nut = self.get_distance(l_s, nut_loc)
                      if dist_man_to_ls != float('inf') and dist_ls_to_nut != float('inf'):
                           cost_pickup_then_walk_to_nut = min(cost_pickup_then_walk_to_nut, dist_man_to_ls + 1 + dist_ls_to_nut)

                 cost_reach_nut_with_spanner = min(cost_reach_nut_with_spanner, cost_pickup_then_walk_to_nut)


            if cost_reach_nut_with_spanner == float('inf'):
                 # This nut cannot be tightened because no usable spanners are reachable
                 return float('inf')

            # Add cost for the tighten action
            nut_cost = cost_reach_nut_with_spanner + 1

            total_h += nut_cost

        # 8. Return the total summed cost.
        return total_h
