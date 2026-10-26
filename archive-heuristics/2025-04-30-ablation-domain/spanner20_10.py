import collections
from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In a real environment, this would be provided by the planner framework.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential errors or unexpected formats
         # print(f"Warning: get_parts received unexpected fact format: {fact}")
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the number of loose goal nuts, the cost to get Bob to the nearest
    location where work is needed (either a nut or a spanner), the cost to pick
    up a spanner if needed, and an estimated cost for moving between subsequent
    nut locations.

    # Assumptions
    - Bob is the only agent who can move, carry, pickup, putdown, and tighten.
    - Bob can carry multiple spanners simultaneously.
    - All goal nuts must be tightened.
    - The location graph defined by 'link' predicates is connected (or relevant parts are).
    - Shortest path distance in the location graph is a reasonable estimate for movement cost.
    - If Bob needs a spanner, there is at least one usable spanner available on the ground somewhere reachable.
    - Any object with the 'usable' predicate is a spanner relevant for tightening nuts.

    # Heuristic Initialization
    - Build a graph of locations based on 'link' predicates to compute shortest path distances.
    - Identify the set of nuts that need to be tightened (goal nuts) from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are in the task's goal conditions and are currently loose in the given state. Let this set be `LooseGoalNuts`.
    2. If `LooseGoalNuts` is empty, the heuristic is 0 (goal reached).
    3. Initialize the heuristic value `h` with the number of loose goal nuts (`len(LooseGoalNuts)`). This accounts for the `tighten` action required for each nut.
    4. Find Bob's current location (`LocBob`) from the state. If Bob's location is not found, the state is likely invalid or unsolvable, return infinity.
    5. Determine if Bob is currently carrying any usable spanner (`BobCarryingUsableSpanner`). This requires checking `carrying` facts and `usable` facts in the state.
    6. Find the locations of all usable spanners that are currently on the ground (`UsableSpannerLocs`). This requires checking `at` facts and `usable` facts in the state, and ensuring the spanner is not carried by Bob.
    7. Find the locations of all loose goal nuts (`LooseGoalNutLocs`). This requires checking `at` facts for the nuts in `LooseGoalNuts`.
    8. Compute shortest path distances from Bob's current location (`LocBob`) to all other reachable locations using BFS on the location graph built during initialization. Store these distances.
    9. Calculate the cost to get a spanner if Bob needs one:
       - If `BobCarryingUsableSpanner` is False:
         - If there are no usable spanners on the ground (`UsableSpannerLocs` is empty), the state is unsolvable (Bob can't get a spanner), return infinity.
         - Find the minimum distance from `LocBob` to any location in `UsableSpannerLocs` using the pre-computed distances. Let this be `min_dist_to_spanner`.
         - If `min_dist_to_spanner` is infinity (no reachable spanner on ground), the state is unsolvable, return infinity.
         - Add `min_dist_to_spanner + 1` to `h`. (+1 for the `pickup` action).
    10. Calculate the cost to get Bob to the nearest loose goal nut location:
        - Find the minimum distance from `LocBob` to any location in `LooseGoalNutLocs` using the pre-computed distances. Let this be `min_dist_to_nut`.
        - If `min_dist_to_nut` is infinity (no reachable nut), the state is unsolvable, return infinity.
        - Add `min_dist_to_nut` to `h`. This accounts for the movement to the first nut location.
    11. Estimate the cost to move between the *remaining* loose goal nuts:
        - If there is more than one loose goal nut (`len(LooseGoalNuts) > 1`):
          - Add `len(LooseGoalNuts) - 1` to `h`. This is a rough estimate assuming one move is needed between each subsequent nut location visit.
    12. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions for tightening nuts.
        - Static facts ('link' relationships).
        - Build the location graph.
        """
        # Store goal conditions for tightening nuts
        self.goal_nuts = {
            get_parts(goal)[1]
            for goal in task.goals
            if match(goal, "tightened", "*")
        }

        # Build the location graph from 'link' facts
        self.location_graph = collections.defaultdict(list)
        locations = set()
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1) # Links are bidirectional
                locations.add(loc1)
                locations.add(loc2)
        self.locations = list(locations) # Store list of all locations

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose goal nuts
        loose_goal_nuts = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "loose", "*") and get_parts(fact)[1] in self.goal_nuts
        }

        # 2. If no loose goal nuts, goal is reached
        if not loose_goal_nuts:
            return 0

        # Initialize heuristic with tighten actions
        h = len(loose_goal_nuts)

        # 4. Find Bob's current location
        bob_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_location = get_parts(fact)[2]
                break
        if bob_location is None:
             # Bob's location is unknown - state is likely invalid
             return float('inf')

        # 8. Compute shortest path distances from Bob's location
        distances_from_bob = {loc: float('inf') for loc in self.locations}
        if bob_location in self.locations: # Ensure Bob's location is in the graph
            queue = collections.deque([(bob_location, 0)])
            visited = {bob_location}
            distances_from_bob[bob_location] = 0

            while queue:
                current_loc, dist = queue.popleft()
                for neighbor in self.location_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances_from_bob[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        # If bob_location is not in self.locations, all distances remain inf.

        def get_dist_from_bob(loc):
             return distances_from_bob.get(loc, float('inf'))


        # 5. Determine if Bob is carrying a usable spanner
        bob_carrying_usable_spanner = False
        carried_spanners = set()
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                spanner = get_parts(fact)[2]
                carried_spanners.add(spanner)

        for spanner in carried_spanners:
             if f"(usable {spanner})" in state:
                  bob_carrying_usable_spanner = True
                  break


        # 6. Find locations of usable spanners on the ground
        usable_spanner_locations_on_ground = set()
        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj, loc = get_parts(fact)[1:]
                  # Check if the object is usable and not carried by Bob
                  # Assuming any usable object is a spanner in this domain context
                  if f"(usable {obj})" in state and obj not in carried_spanners:
                       usable_spanner_locations_on_ground.add(loc)


        # 9. Cost to get a spanner if needed
        spanner_needed = not bob_carrying_usable_spanner
        if spanner_needed:
             if not usable_spanner_locations_on_ground:
                  # Spanner needed but none available on the ground
                  # Assuming this implies unsolvability if nuts need tightening
                  return float('inf')

             min_dist_to_spanner = float('inf')
             for loc in usable_spanner_locations_on_ground:
                  dist = get_dist_from_bob(loc)
                  if dist < min_dist_to_spanner:
                       min_dist_to_spanner = dist

             if min_dist_to_spanner == float('inf'):
                  # Spanner needed but no usable spanner reachable on ground
                  return float('inf') # Unsolvable state
             else:
                  h += min_dist_to_spanner + 1 # +1 for pickup action


        # 7. Find locations of loose goal nuts
        loose_goal_nut_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in loose_goal_nuts:
                    loose_goal_nut_locations.add(loc)

        # 10. Cost to get Bob to the nearest loose goal nut location
        min_dist_to_nut = float('inf')
        if loose_goal_nut_locations:
             for loc in loose_goal_nut_locations:
                  dist = get_dist_from_bob(loc)
                  if dist < min_dist_to_nut:
                       min_dist_to_nut = dist

             if min_dist_to_nut == float('inf'):
                  # Loose goal nuts exist but are unreachable
                  return float('inf') # Unsolvable state
             else:
                  h += min_dist_to_nut # Movement to the first nut
        # else: # This case is caught by the initial check `if not loose_goal_nuts:`
        #      pass


        # 11. Estimate cost to move between remaining loose goal nuts
        # After reaching the first nut, Bob needs to visit len(loose_goal_nuts) - 1 more nuts.
        # This is a rough estimate assuming one move per additional nut.
        if len(loose_goal_nuts) > 1:
             h += (len(loose_goal_nuts) - 1)


        return h
