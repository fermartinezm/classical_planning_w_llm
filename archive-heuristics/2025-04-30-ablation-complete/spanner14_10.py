# from heuristics.heuristic_base import Heuristic # Assuming this base class exists
from fnmatch import fnmatch
from collections import deque

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, though state facts should be consistent
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the domain-dependent heuristic class
# class spannerHeuristic(Heuristic): # Inherit from Heuristic if available
class spannerHeuristic: # Provide the class definition directly
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose goal nuts.
    It simulates a greedy strategy where the man always moves towards the closest required item:
    either a usable spanner (if not carrying one) or a loose goal nut (if carrying a usable spanner).
    The estimated cost is the sum of walk actions, pickup actions, and tighten actions
    required to complete the simulated sequence.

    # Assumptions
    - There is only one man.
    - The man can carry only one spanner at a time.
    - Spanners are consumed after one use (tightening a nut).
    - The goal is always to tighten a specific set of nuts.
    - All locations mentioned in 'link' facts form a connected graph.
    - All objects (man, spanners, nuts) are initially located at one of these locations.

    # Heuristic Initialization
    - Identify all locations by parsing 'link' facts from the static information.
    - Build a graph representing the connections between locations based on 'link' facts.
    - Precompute shortest path distances between all pairs of identified locations using BFS.
    - Store the set of nuts that are specified in the goal conditions.
    - Store the mapping of object names to types from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Extract the man object, his current location, and determine if he is carrying a usable spanner.
    2. Identify the set of nuts that are part of the goal and are currently in a 'loose' state. Map these loose goal nuts to their current locations.
    3. Identify the set of usable spanners that are currently 'at' a location (i.e., not carried by the man) and store their locations.
    4. Check if the total number of available usable spanners (those at locations plus the one the man might be carrying) is less than the number of loose goal nuts. If insufficient spanners exist, the problem is unsolvable from this state, and the heuristic returns infinity.
    5. Initialize the estimated total cost to 0.
    6. Simulate a greedy plan execution until all loose goal nuts are considered 'tightened' in the simulation:
        a. If the man is currently simulated as carrying a usable spanner:
            - Find the loose goal nut that is closest (in terms of walk distance) to the man's current simulated location.
            - Add the walk distance to this nut's location to the total cost.
            - Add 1 to the cost for the 'tighten_nut' action.
            - Update the man's current simulated location to the nut's location.
            - The spanner is considered consumed; the man is no longer simulated as carrying a usable spanner.
            - Remove the chosen nut from the set of remaining loose goal nuts to be tightened.
        b. If the man is currently simulated as not carrying a usable spanner:
            - Find the location of a usable spanner that is closest to the man's current simulated location among the available spanner locations.
            - Add the walk distance to this spanner location to the total cost.
            - Add 1 to the cost for the 'pickup_spanner' action.
            - Update the man's current simulated location to the spanner location.
            - The man is now simulated as carrying a usable spanner.
            - Remove the chosen spanner location from the set of available spanner locations for future pickups.
        c. If at any point the closest required item (nut or spanner) is unreachable from the man's current simulated location (distance is infinity), the state is likely unsolvable, and the heuristic returns infinity.
    7. Once all loose goal nuts have been processed in the simulation, return the accumulated total cost as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and storing goal nuts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.task_objects = task.objects # Store objects to identify types

        # Identify all locations from link facts
        locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)

        self.locations = list(locations) # Use a list for consistent ordering if needed, though set is fine for iteration

        # Build the location graph from link facts
        self.graph = {loc: set() for loc in self.locations}
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations are part of the graph we are building
                if loc1 in self.graph and loc2 in self.graph:
                    self.graph[loc1].add(loc2)
                    self.graph[loc2].add(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify goal nuts from the task goals
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Find the man object (assuming there's exactly one man)
        self.man_obj = None
        for obj, obj_type in self.task_objects.items():
            if obj_type == 'man':
                self.man_obj = obj
                break

    def _bfs(self, start_loc):
        """Perform BFS from start_loc to find distances to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}

        # Only proceed if the start location is part of the graph we built
        if start_loc in self.graph:
            distances[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                current_loc = queue.popleft()

                # Check if current_loc is a valid key in the graph
                if current_loc not in self.graph:
                    continue # Should not happen if locations come from graph keys

                for neighbor in self.graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        # If start_loc is not in self.graph, all distances remain infinity (except potentially start_loc itself if added)
        # This correctly models isolation.

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract current state information ---
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_obj, "*"):
                man_loc = get_parts(fact)[2]
                break
        # If man_loc is None, the state is malformed. Return infinity.
        if man_loc is None:
             return float('inf')

        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_obj, "*"):
                carried_spanner = get_parts(fact)[2]
                break

        is_carrying_usable = carried_spanner is not None and f"(usable {carried_spanner})" in state

        # Identify loose goal nuts and their locations
        loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}

        nut_locations = {}
        for nut in loose_goal_nuts:
             found_loc = False
             for fact in state:
                 if match(fact, "at", nut, "*"):
                     nut_locations[nut] = get_parts(fact)[2]
                     found_loc = True
                     break
             # If a loose goal nut is not 'at' any location, it's likely unsolvable
             if not found_loc:
                 return float('inf')


        # Find usable spanners that are currently 'at' a location (not carried)
        # First find all usable spanner objects
        usable_spanner_objects = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # Then find locations of usable spanners that are currently 'at' a location
        available_spanner_locs = set()
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj_at_loc = get_parts(fact)[1]
                 loc = get_parts(fact)[2]
                 if obj_at_loc in usable_spanner_objects:
                     # This spanner is usable and at a location.
                     # Objects being carried do not have an 'at' fact, so this is correct.
                     available_spanner_locs.add(loc)


        # Check solvability based on spanner count
        required_spanners = len(loose_goal_nuts)
        available_spanners_count = len(available_spanner_locs) + (1 if is_carrying_usable else 0)

        if required_spanners > available_spanners_count:
            return float('inf') # Not enough spanners exist

        # If no loose goal nuts, goal is reached for these nuts
        if not loose_goal_nuts:
            return 0

        # --- Greedy Simulation ---
        current_man_loc_sim = man_loc
        current_carrying_usable_sim = is_carrying_usable
        remaining_nuts_sim = set(loose_goal_nuts)
        available_spanner_locations_sim = set(available_spanner_locs) # Use a copy for simulation

        cost = 0

        while remaining_nuts_sim:
            if current_carrying_usable_sim:
                # Man has a spanner, go tighten the closest remaining nut
                closest_nut = None
                min_dist_to_nut = float('inf')
                target_nut_loc = None

                for nut in remaining_nuts_sim:
                    loc = nut_locations[nut]
                    # Check if the current man location or target nut location is unknown in distances
                    # This handles cases where locations might be isolated or not in the graph
                    if current_man_loc_sim not in self.distances or loc not in self.distances[current_man_loc_sim]:
                         return float('inf') # Unreachable location

                    dist = self.distances[current_man_loc_sim][loc]
                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        closest_nut = nut
                        target_nut_loc = loc

                # If min_dist_to_nut is still infinity, no remaining nut is reachable
                if min_dist_to_nut == float('inf'):
                     return float('inf')

                cost += min_dist_to_nut # Walk actions
                cost += 1 # Tighten action
                current_man_loc_sim = target_nut_loc
                current_carrying_usable_sim = False # Spanner consumed
                remaining_nuts_sim.remove(closest_nut)

            else:
                # Man needs a spanner. Go get the closest available one.
                closest_spanner_loc = None
                min_dist_to_spanner = float('inf')

                for s_loc in available_spanner_locations_sim:
                    # Check if the current man location or target spanner location is unknown in distances
                    # This handles cases where locations might be isolated or not in the graph
                    if current_man_loc_sim not in self.distances or s_loc not in self.distances[current_man_loc_sim]:
                         return float('inf') # Unreachable location

                    dist = self.distances[current_man_loc_sim][s_loc]
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        closest_spanner_loc = s_loc

                # If min_dist_to_spanner is still infinity, no available spanner is reachable
                if min_dist_to_spanner == float('inf'):
                    return float('inf')

                cost += min_dist_to_spanner # Walk actions
                cost += 1 # Pickup action
                current_man_loc_sim = closest_spanner_loc
                current_carrying_usable_sim = True
                available_spanner_locations_sim.remove(closest_spanner_loc)

        return cost
