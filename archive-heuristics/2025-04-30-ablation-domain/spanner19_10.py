from fnmatch import fnmatch
from collections import deque
import math
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic


# Helper functions get_parts and match from Logistics example
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
    # The zip approach from examples implicitly handles different lengths
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Inherit from Heuristic in the actual planning system
# class spannerHeuristic(Heuristic):
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It sums the cost of the 'tighten' action for each loose goal nut
    and the estimated movement/pickup cost for Bob to reach all necessary
    locations with a usable spanner. The movement cost is estimated using
    the maximum shortest distance to any required location, which is a simple
    and efficient relaxation.

    # Assumptions
    - Bob can carry multiple spanners (based on example state).
    - Only usable spanners can be picked up or used for tightening.
    - The cost of movement between linked locations is 1.
    - The cost of pickup, drop, and tighten actions is 1.
    - The heuristic uses the maximum distance to any required location as a
      lower bound for the movement cost to visit multiple locations.
    - Loose nuts are always located at a specific location on the ground.
    - Usable spanners are either carried by Bob or located at a specific location on the ground.

    # Heuristic Initialization
    - Extracts all relevant locations from the initial state and static facts (links).
    - Builds an undirected graph representing the links between locations.
    - Computes all-pairs shortest paths (distances) between all identified locations using BFS.
    - Stores the goal conditions to identify which nuts ultimately need tightening.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Goal Nuts:** Determine the set of nuts that must be tightened according to the task's goal conditions.
    2.  **Identify Loose Goal Nuts:** From the current state, find which of the goal nuts are currently in a 'loose' state. These are the nuts that still need work.
    3.  **Check for Goal State Subset:** If the set of loose goal nuts is empty, it means all goal nuts are already tightened. The heuristic value is 0.
    4.  **Locate Bob:** Find Bob's current location in the state. If Bob's location is not found, the state is invalid or unsolvable, return infinity.
    5.  **Locate Loose Goal Nuts:** For each loose goal nut, find its current location in the state. Collect the set of unique locations where loose goal nuts are found. If any loose goal nut's location cannot be found, return infinity.
    6.  **Check for Usable Spanner:** Determine if Bob is currently carrying any spanner that is marked as 'usable' in the state.
    7.  **Locate Usable Ground Spanners:** Find all usable spanners that are currently located on the ground (not carried by Bob) and their respective locations.
    8.  **Calculate Base Cost:** The minimum number of 'tighten' actions required is equal to the number of loose goal nuts. Initialize the heuristic value with this count.
    9.  **Calculate Movement and Acquisition Cost:** This is the cost to get Bob, equipped with a usable spanner, to all locations where loose goal nuts are situated.
        a.  If the set of loose goal nut locations is empty (already handled in step 3), the movement cost is 0.
        b.  If there are loose goal nut locations:
            i.  **If Bob is carrying a usable spanner:** Bob already has the necessary tool. The movement cost is the maximum shortest distance from Bob's current
                location to any location with a loose goal nut. This estimates the longest single trip needed. If any required nut location is unreachable, return infinity.
            ii. **If Bob is not carrying a usable spanner:** Bob must first acquire one. Find the usable spanner on the ground that is reachable and closest to Bob's
                current location. If no reachable usable ground spanner exists, return infinity. The cost to acquire the spanner is the distance to this nearest spanner + 1 (for the 'pickup' action). After picking up the spanner, Bob needs to travel to the loose goal nut locations. The additional movement cost is the maximum shortest distance from the location where the spanner was picked up to any of the loose goal nut locations. If any required nut location is unreachable from the spanner location, return infinity. The total movement/acquisition cost is the sum of the spanner acquisition cost and this subsequent maximum travel cost.
    10. **Sum Costs:** Add the calculated movement and acquisition cost to the base cost (from step 8).
    11. **Return Heuristic Value:** Return the total calculated cost. If any step resulted in infinity due to unreachability or missing information, that infinity value is propagated.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing distances between locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Identify all relevant locations
        all_locations = set()
        self.location_graph = {} # Adjacency list for locations

        # Locations from links
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        # Locations from initial state 'at' facts
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3: # Ensure it's (at obj loc)
                    obj, loc = parts[1], parts[2]
                    all_locations.add(loc)
                    self.location_graph.setdefault(loc, set()) # Ensure all locations are keys

        # Locations from goal state 'at' facts (spanner goals are 'tightened', but include for generality)
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 parts = get_parts(goal)
                 if len(parts) == 3: # Ensure it's (at obj loc)
                    obj, loc = parts[1], parts[2]
                    all_locations.add(loc)
                    self.location_graph.setdefault(loc, set()) # Ensure all locations are keys


        self.all_locations = list(all_locations) # Store as list

        # 2. Compute all-pairs shortest paths (distances) using BFS
        self.distances = {}
        for start_node in self.all_locations:
            self.distances[start_node] = {}
            queue = deque([start_node])
            dist = {start_node: 0}
            self.distances[start_node][start_node] = 0

            while queue:
                u = queue.popleft()
                current_dist = dist[u]

                # Check if u is in the graph (it should be if it's in all_locations)
                if u in self.location_graph:
                    for v in self.location_graph[u]:
                        if v not in dist:
                            dist[v] = current_dist + 1
                            self.distances[start_node][v] = current_dist + 1
                            queue.append(v)

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returning infinity if unreachable."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
            return math.inf
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Goal Nuts
        goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # 2. Identify Loose Goal Nuts and their locations
        loose_goal_nuts = set()
        nut_location_map = {} # Map nut to its current location

        # Find locations of all nuts in the state
        for fact in state:
            if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3: # Ensure it's (at obj loc)
                    obj, loc = parts[1], parts[2]
                    # Check if the object is a nut (simple check based on naming convention)
                    if obj.startswith("nut"):
                        nut_location_map[obj] = loc

        # Check which goal nuts are loose
        for nut in goal_nuts:
            if f"(loose {nut})" in state:
                 loose_goal_nuts.add(nut)

        # 3. Check for Goal State Subset
        if not loose_goal_nuts:
            return 0 # All goal nuts are tightened

        # 4. Locate Bob
        bob_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_location = get_parts(fact)[2]
                break
        if bob_location is None:
             # Bob's location not found? Invalid state.
             return math.inf

        # 5. Locate Loose Goal Nuts Locations
        loose_goal_nut_locations = set()
        for nut in loose_goal_nuts:
             if nut in nut_location_map:
                 loose_goal_nut_locations.add(nut_location_map[nut])
             else:
                 # A loose goal nut has no location in the state? Invalid state.
                 return math.inf

        # If loose_goal_nut_locations is empty, it means loose_goal_nuts was empty,
        # which is handled at the start. So, if we reach here, it's not empty.

        # 6. Check if Bob is carrying a usable spanner
        bob_has_usable_spanner = False
        usable_spanners_on_ground = [] # List of (spanner_name, location)
        all_usable_spanners = set() # Set of usable spanner names

        # First find all usable spanners
        for fact in state:
             if match(fact, "usable", "*"):
                 all_usable_spanners.add(get_parts(fact)[1])

        # Check if Bob is carrying any of them or if they are on the ground
        for spanner in all_usable_spanners:
            if f"(carrying bob {spanner})" in state:
                bob_has_usable_spanner = True
                # If Bob has one, we don't need to look for ground spanners for acquisition
                usable_spanners_on_ground = [] # Clear the list
                break # Found one, no need to check others for 'carrying'
            else:
                # Check if this usable spanner is on the ground
                for fact in state:
                    if match(fact, "at", spanner, "*"):
                        loc = get_parts(fact)[2]
                        usable_spanners_on_ground.append((spanner, loc))
                        break # Found location for this spanner, move to next usable spanner

        # 8. Calculate base cost (tighten actions)
        h = len(loose_goal_nuts)

        # 9. Calculate Movement and Acquisition Cost
        movement_cost = 0

        # loose_goal_nut_locations is guaranteed not empty here

        if bob_has_usable_spanner:
            # Bob has a spanner, just need to reach the nut locations
            max_dist_to_nut_locs = 0
            for nut_loc in loose_goal_nut_locations:
                dist = self.get_distance(bob_location, nut_loc)
                if dist == math.inf:
                    # Cannot reach a required nut location
                    return math.inf
                max_dist_to_nut_locs = max(max_dist_to_nut_locs, dist)
            movement_cost = max_dist_to_nut_locs
        else:
            # Bob needs to acquire a spanner first
            min_dist_to_ground_spanner = math.inf
            nearest_spanner_loc = None

            for spanner_name, spanner_loc in usable_spanners_on_ground:
                dist = self.get_distance(bob_location, spanner_loc)
                if dist < min_dist_to_ground_spanner:
                    min_dist_to_ground_spanner = dist
                    nearest_spanner_loc = spanner_loc

            if nearest_spanner_loc is None or min_dist_to_ground_spanner == math.inf:
                # No reachable usable spanner on the ground, and Bob isn't carrying one
                return math.inf # Problem likely unsolvable

            # Cost to get to the nearest spanner + pickup
            cost_to_acquire_spanner = min_dist_to_ground_spanner + 1

            # Cost to travel from the nearest spanner location to the nut locations
            max_dist_from_spanner_to_nut_locs = 0
            for nut_loc in loose_goal_nut_locations:
                dist = self.get_distance(nearest_spanner_loc, nut_loc)
                if dist == math.inf:
                     # Cannot reach a required nut location from the spanner location
                     return math.inf
                max_dist_from_spanner_to_nut_locs = max(max_dist_from_spanner_to_nut_locs, dist)

            movement_cost = cost_to_acquire_spanner + max_dist_from_spanner_to_nut_locs

        # 10. Sum costs
        h += movement_cost

        # 11. Return heuristic value
        return h
