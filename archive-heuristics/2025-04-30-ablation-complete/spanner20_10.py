from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class exists and has __init__(self, task) and __call__(self, node)
# from heuristics.heuristic_base import Heuristic
# If running standalone for testing or if the base class is not provided,
# you might need a dummy definition like:
class Heuristic:
    def __init__(self, task): pass
    def __call__(self, node): pass


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to find the nearest location from a source location within a set of target locations
def find_nearest_loc(source_loc, target_locs, dist_matrix):
    min_d = float('inf')
    nearest = None
    # target_locs can be a dict {name: loc} or a set/list of locs
    if isinstance(target_locs, dict):
        locations = target_locs.values()
    else:
        locations = target_locs

    for target_loc in locations:
        if source_loc in dist_matrix and target_loc in dist_matrix[source_loc]: # Check if reachable
             if dist_matrix[source_loc][target_loc] < min_d:
                  min_d = dist_matrix[source_loc][target_loc]
                  nearest = target_loc
    return nearest # Returns None if target_locs is empty or no location is reachable


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It counts the minimum required tighten and pickup actions and estimates the
    travel cost based on the distance to the first required location and an
    approximation for subsequent travel.

    # Assumptions:
    - Each loose nut requires one tighten action and consumes one usable spanner.
    - The man can only carry one spanner at a time.
    - The man must travel to spanner locations to pick them up and to nut locations to tighten them.
    - The cost of walking between any two directly linked locations is 1.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Identify the name of the man object by looking for the object involved in 'carrying' or the only locatable object not identified as a nut or spanner from initial facts.
    - Build a graph of locations based on `link` facts.
    - Compute all-pairs shortest paths between connected locations using BFS.
    - Calculate the average shortest distance between any two connected locations that are reachable from each other.
    - Store goal nuts for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify the locations of all loose nuts that are goal conditions.
    3. Identify the locations of all usable spanners on the ground.
    4. Determine if the man is currently carrying a usable spanner.
    5. Count the number of loose goal nuts (`N_loose`). If 0, heuristic is 0.
    6. If there are loose goal nuts but not enough usable spanners available (carried + on ground) in the current state, return infinity (unsolvable from here).
    7. Calculate the minimum number of `pickup_spanner` actions needed: `max(0, N_loose - (1 if man is carrying a usable spanner else 0))`.
    8. The base heuristic cost is the sum of minimum tighten actions (`N_loose`) and minimum pickup actions.
    9. Estimate the walk cost:
       a. Cost for the first travel segment:
          - If the man is not carrying a usable spanner, the first necessary step is to go to the nearest reachable usable spanner location, then from there to the nearest reachable loose goal nut location. Calculate the sum of these two shortest distances.
          - If the man is carrying a usable spanner, the first necessary step is to go directly to the nearest reachable loose goal nut location. Calculate this shortest distance.
          - If any required location is unreachable, return infinity.
       b. Cost for subsequent travel segments:
          - For the remaining `N_loose - 1` nuts, the man needs to travel between locations to get new spanners and reach the next nut.
          - Approximate the cost of these subsequent walks by multiplying the number of remaining nuts (`N_loose - 1`) by twice the average distance between locations (representing a trip from a nut area to a spanner area and back to a nut area). This is a rough estimate.

    10. Sum the base cost, the first travel cost, and the subsequent travel cost estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by pre-calculating distances and identifying objects."""
        # self.goals = task.goals # Stored in parent Heuristic class? Let's access directly from task.
        static_facts = task.static
        initial_state = task.initial_state

        # Identify object names based on predicates in initial state/goals/static
        self.man_name = None
        self.nut_names = set()
        self.spanner_names = set()
        self.locations = set()

        all_relevant_facts = initial_state | static_facts | task.goals

        known_locatables = set()
        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if parts[0] in ['loose', 'tightened'] and len(parts) == 2:
                self.nut_names.add(parts[1])
                known_locatables.add(parts[1])
            elif parts[0] == 'usable' and len(parts) == 2:
                self.spanner_names.add(parts[1])
                known_locatables.add(parts[1])
            elif parts[0] == 'carrying' and len(parts) == 3:
                # The first arg is the man, second is a spanner
                self.man_name = parts[1]
                self.spanner_names.add(parts[2])
                known_locatables.add(parts[1])
                known_locatables.add(parts[2])
            elif parts[0] == 'at' and len(parts) == 3:
                 self.locations.add(parts[2])
            elif parts[0] == 'link' and len(parts) == 3:
                 self.locations.add(parts[1])
                 self.locations.add(parts[2])


        # If man_name wasn't found via 'carrying', find the locatable that isn't a nut/spanner
        if self.man_name is None:
            for fact in all_relevant_facts:
                parts = get_parts(fact)
                if parts[0] == 'at' and len(parts) == 3:
                    obj = parts[1]
                    if obj not in known_locatables:
                        self.man_name = obj
                        known_locatables.add(obj)
                        break # Found the man

        if self.man_name is None:
             # Fallback or error if man not found - should not happen in valid problems
             # print("Warning: Man object name not found. Assuming 'bob'.")
             self.man_name = 'bob' # Last resort fallback


        # Build location graph from link facts
        self.adj = {} # Adjacency list: location -> [neighbor1, neighbor2, ...]
        for loc in self.locations:
            self.adj[loc] = []

        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                if l1 in self.locations and l2 in self.locations: # Ensure locations are known
                    self.adj[l1].append(l2)
                    self.adj[l2].append(l1) # Links are bidirectional
                # else: print(f"Warning: Link fact with unknown location: {fact}")


        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.dist[start_node][start_node] = 0

            while q:
                curr_node, d = q.popleft()

                for neighbor in self.adj.get(curr_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start_node][neighbor] = d + 1
                        q.append((neighbor, d + 1))

        # Calculate average shortest distance between connected locations
        reachable_pairs_dist_sum = 0
        reachable_pairs_count = 0
        for l1 in self.locations:
            if l1 in self.dist: # Ensure BFS started successfully from l1
                for l2 in self.dist[l1]: # Iterate over reachable locations from l1
                    if l1 != l2:
                         reachable_pairs_dist_sum += self.dist[l1][l2]
                         reachable_pairs_count += 1

        self.avg_dist = (reachable_pairs_dist_sum / reachable_pairs_count) if reachable_pairs_count > 0 else 1 # Avoid division by zero


        # Store goal nuts for quick lookup (names only)
        self.goal_nuts = set()
        for goal in task.goals: # Access goals from task object
             parts = get_parts(goal)
             if parts[0] == 'tightened' and len(parts) == 2:
                 self.goal_nuts.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                 man_loc = get_parts(fact)[2]
                 break

        if man_loc is None:
             # Man's location must be in the state if it's a valid state
             return float('inf') # Should not happen


        # 2. Identify locations of loose nuts that are goals
        loose_goal_nut_locs = {} # Map nut name to location for loose nuts that are goals
        current_tightened_nuts = set()
        for fact in state:
             if match(fact, "tightened", "*"):
                  current_tightened_nuts.add(get_parts(fact)[1])

        nuts_to_tighten = self.goal_nuts - current_tightened_nuts

        if not nuts_to_tighten:
             return 0 # All goal nuts are tightened

        for nut_name in nuts_to_tighten:
             # Find location of this nut
             found_loc = False
             for loc_fact in state:
                 if match(loc_fact, "at", nut_name, "*"):
                     loose_goal_nut_locs[nut_name] = get_parts(loc_fact)[2]
                     found_loc = True
                     break
             if not found_loc:
                  # Goal nut not found at any location in the state? Problematic state.
                  return float('inf') # Should not happen


        # 3. Identify locations of usable spanners on the ground
        usable_spanner_locs = {} # Map spanner name to location
        for fact in state:
            if match(fact, "usable", "*"):
                spanner_name = get_parts(fact)[1]
                # Check if this spanner is on the ground (not carried)
                is_carried = False
                for carry_fact in state:
                    if match(carry_fact, "carrying", self.man_name, spanner_name):
                        is_carried = True
                        break
                if not is_carried:
                    # Find location of this spanner
                    found_loc = False
                    for loc_fact in state:
                        if match(loc_fact, "at", spanner_name, "*"):
                            usable_spanner_locs[spanner_name] = get_parts(loc_fact)[2]
                            found_loc = True
                            break
                    # If usable spanner is not carried and not at a location, it's lost?
                    # Assume usable spanners are either carried or at a location.
                    # If not found_loc, it's a state inconsistency or the spanner is lost.
                    # For heuristic, assume it's not available on the ground if location not found.


        # 4. Determine if the man is currently carrying a usable spanner
        carrying_usable = False
        for fact in state:
             if match(fact, "carrying", self.man_name, "*"):
                 carried_spanner_name = get_parts(fact)[2]
                 # Check if the carried spanner is usable
                 for usable_fact in state:
                     if match(usable_fact, "usable", carried_spanner_name):
                         carrying_usable = True
                         break
                 break # Man can only carry one spanner


        # 5. Count the number of loose nuts that are goals
        N_loose = len(nuts_to_tighten)
        # Already handled N_loose == 0 case above

        # 6. Check solvability regarding spanners
        # Check against usable spanners currently accessible (carried + on ground)
        num_usable_in_state = (1 if carrying_usable else 0) + len(usable_spanner_locs)
        if N_loose > num_usable_in_state:
             # Not enough usable spanners *currently accessible* to tighten all remaining nuts.
             return float('inf')


        # 7. Calculate minimum pickup actions needed
        num_pickups_needed = max(0, N_loose - (1 if carrying_usable else 0))

        # 8. Base cost: tighten + pickup actions
        cost = N_loose + num_pickups_needed

        # 9. Estimate walk cost
        walk_cost = 0

        # Find nearest reachable loose goal nut location
        nearest_nut_loc = find_nearest_loc(man_loc, loose_goal_nut_locs, self.dist)

        # Find nearest reachable usable spanner location on the ground
        nearest_spanner_loc = find_nearest_loc(man_loc, usable_spanner_locs, self.dist)

        # Check if required locations are reachable from man_loc
        if nearest_nut_loc is None: # Loose goal nuts exist but none are reachable
             return float('inf')
        # If pickups are needed, check if a spanner is reachable
        if num_pickups_needed > 0 and nearest_spanner_loc is None:
             # Need pickups but no reachable usable spanners on ground
             return float('inf')


        # a. Cost for the first travel segment:
        first_segment_walk = 0
        if not carrying_usable:
            # Man needs to go to nearest spanner first, then to nearest nut from there.
            # We already checked reachability from man_loc to nearest_spanner_loc.
            # Now check reachability from nearest_spanner_loc to nearest_nut_loc.
            if nearest_spanner_loc in self.dist and nearest_nut_loc in self.dist[nearest_spanner_loc]:
                 first_segment_walk = self.dist[man_loc][nearest_spanner_loc] + self.dist[nearest_spanner_loc][nearest_nut_loc]
            else:
                 # This case should be covered by the check that nearest_nut_loc is reachable from man_loc
                 # if the graph is connected, but adding for safety.
                 return float('inf')
        else:
            # Man is carrying spanner, goes directly to nearest nut
            # We already checked reachability from man_loc to nearest_nut_loc.
            first_segment_walk = self.dist[man_loc][nearest_nut_loc]

        walk_cost += first_segment_walk

        # b. Cost for subsequent travel segments:
        # For the remaining N_loose - 1 nuts, the man needs to travel.
        # Each remaining nut requires getting a spanner (walk to spanner, pickup)
        # and traveling to the nut (walk to nut, tighten).
        # The travel involves going from a nut location to a spanner location, then from
        # that spanner location to the next nut location.
        # Number of such full cycles (nut -> spanner -> nut) is N_loose - 1.
        # Each cycle involves roughly 2 walk segments.
        # Approximate cost per segment by average distance.
        if N_loose > 1:
             # Number of subsequent nut-spanner-nut cycles
             num_subsequent_cycles = N_loose - 1
             # Each cycle involves roughly 2 walk segments (nut->spanner, spanner->nut)
             walk_cost += num_subsequent_cycles * 2 * self.avg_dist


        # 10. Sum costs
        cost += walk_cost

        return cost
