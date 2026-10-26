from fnmatch import fnmatch
import collections
import math
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic function for the Spanner domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing the number of
    loose nuts that need tightening (base cost for tighten actions) and the estimated
    minimum cost to make *one* of these loose nuts ready for tightening. Making a nut
    ready involves getting the man to the nut's location while carrying a usable spanner.

    # Assumptions
    - The goal is to achieve `(tightened ?n)` for a specific set of nuts.
    - A spanner becomes unusable after one `tighten_nut` action.
    - Travel cost between linked locations is 1.
    - Pickup cost is 1. Tighten cost is 1.
    - There is exactly one man.

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task's goal conditions.
    - Builds an undirected graph of locations based on `link` predicates found in the static facts.
    - Computes all-pairs shortest paths between locations using Breadth-First Search (BFS) and stores them.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man object and his current location from the state.
    2. Identify the set of nuts that are specified in the goal but are currently in a `loose` state (`LooseGoalNuts`).
    3. If `LooseGoalNuts` is empty, the current state is a goal state, so the heuristic value is 0.
    4. Identify all usable spanners currently available in the state (either carried by the man and usable, or located on the ground and usable).
    5. If the number of `LooseGoalNuts` is greater than the total number of available usable spanners, the problem is unsolvable from this state under the domain rules (spanners don't become usable again), so return `math.inf`.
    6. Initialize the heuristic cost with the number of `LooseGoalNuts`. This represents the minimum number of `tighten_nut` actions required.
    7. Calculate the minimum cost required to make *any* single nut from `LooseGoalNuts` ready to be tightened. This involves:
       - Getting the man to the location of the chosen nut.
       - Ensuring the man is carrying a usable spanner.
       - If the man is already carrying a usable spanner, the cost is just the travel to the nut's location.
       - If the man is not carrying a usable spanner, he must acquire one. The cost is estimated by considering two sub-options for getting a spanner and reaching the nut:
         a) Travel to the nut's location first, then acquire a spanner (either pick one up if available at the nut's location, or travel from the nut's location to the closest usable spanner and pick it up).
         b) Acquire the closest usable spanner first (travel to spanner, pickup), then travel from the spanner's location to the nut's location.
       - The minimum cost among all loose goal nuts and all spanner acquisition options is found.
    8. Add the minimum cost calculated in step 7 to the base heuristic cost from step 6.
    9. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal nuts, building the location graph, and computing distances."""
        self.goals = task.goals
        self.static_facts = task.static

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "tightened":
                self.goal_nuts.add(parts[1])

        # Build location graph from link facts
        self.location_graph = collections.defaultdict(set)
        self.all_locations = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[(start_loc, start_loc)] = 0
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()

                for neighbor in self.location_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations, or math.inf if unreachable."""
        # Ensure both locations are part of the known graph
        if loc1 not in self.all_locations or loc2 not in self.all_locations:
             return math.inf
        return self.distances.get((loc1, loc2), math.inf)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant information from the current state
        man_name = None
        man_location = None
        carried_spanner = None
        is_carrying_usable_spanner = False

        current_usable_spanners_on_ground = set()
        current_loose_nuts = set()
        current_nut_locations = {} # Map nut name to location
        current_spanner_locations = {} # Map spanner name to location
        all_spanners_in_state = set() # Track all spanners seen

        # First pass to identify man (via carrying) and collect nuts/spanners and their locations/states
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "carrying":
                 man_name = parts[1]
                 carried_spanner = parts[2]

            # Collect locations and identify nuts/spanners
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                # Simple check if obj is likely a nut or spanner based on predicate names
                # This is a heuristic within the heuristic due to lack of type info
                is_nut_like = any(match(f, "loose", obj) or match(f, "tightened", obj) for f in state)
                is_spanner_like = any(match(f, "usable", obj) or match(f, "carrying", "*", obj) for f in state)

                if is_nut_like:
                    current_nut_locations[obj] = loc
                elif is_spanner_like:
                    current_spanner_locations[obj] = loc
                    all_spanners_in_state.add(obj)
                # If it's neither nut-like nor spanner-like, and we haven't found the man yet via 'carrying', assume it's the man.
                # This is a fallback and might be incorrect if other object types exist.
                # A more robust way would be to parse object types from the PDDL problem file.
                # Given the example domains, this simple check is likely sufficient.
                elif man_name is None:
                     man_name = obj # Assume this is the man
                     man_location = loc # And this is his location

            # Identify loose nuts
            if parts[0] == "loose":
                 current_loose_nuts.add(parts[1])

        # Second pass to confirm man's location if not found in first pass (e.g., man not carrying anything)
        if man_name is not None and man_location is None:
             for fact in state:
                 if match(fact, "at", man_name, "*"):
                     man_location = get_parts(fact)[2]
                     break

        # Check if carried spanner is usable
        if carried_spanner is not None:
             if any(match(f, "usable", carried_spanner) for f in state):
                 is_carrying_usable_spanner = True

        # Identify usable spanners on the ground
        for spanner in all_spanners_in_state:
             if spanner != carried_spanner: # Only consider spanners not carried
                 if any(match(f, "usable", spanner) for f in state):
                     # Check if it's on the ground (has an 'at' predicate)
                     if any(match(f, "at", spanner, "*") for f in state):
                         current_usable_spanners_on_ground.add(spanner)


        if man_name is None or man_location is None:
             # Could not identify the man or his location. State is likely malformed for this heuristic.
             # print(f"Error: Could not identify man or his location in state: {state}")
             return math.inf


        # 2. Identify LooseGoalNuts
        loose_goal_nuts = {n for n in self.goal_nuts if n in current_loose_nuts}

        # 3. If LooseGoalNuts is empty, the heuristic is 0.
        if not loose_goal_nuts:
            return 0

        # 4. Identify all usable spanners (carried or on ground).
        # We need the count for the unsolvable check.
        total_usable_spanners_count = len(current_usable_spanners_on_ground) + (1 if is_carrying_usable_spanner else 0)


        # 5. If the number of LooseGoalNuts exceeds the number of available usable spanners, return infinity.
        if len(loose_goal_nuts) > total_usable_spanners_count:
             # print(f"Unsolvable: Need {len(loose_goal_nuts)} spanners but only {total_usable_spanners_count} available.")
             return math.inf

        # 6. Base heuristic cost is the number of tighten_nut actions.
        cost = len(loose_goal_nuts)

        # 7. Calculate the minimum cost to make any single loose goal nut ready for tightening.
        min_cost_one_nut_ready = math.inf

        for nut in loose_goal_nuts:
            nut_location = current_nut_locations.get(nut)
            if nut_location is None:
                 # Nut location not found in state. Problematic state.
                 # print(f"Error: Location not found for nut {nut} in state: {state}")
                 return math.inf

            cost_to_reach_nut_loc = self.get_distance(man_location, nut_location)

            if cost_to_reach_nut_loc == math.inf:
                 # Cannot reach the nut's location. Unsolvable.
                 return math.inf

            if is_carrying_usable_spanner:
                # Man has a usable spanner, just needs to get to the nut.
                cost_to_make_this_nut_ready = cost_to_reach_nut_loc
            else:
                # Man needs to get a spanner first.
                # Option A: Go to nut first, then get spanner.
                cost_get_spanner_from_nut_loc = math.inf
                usable_spanner_at_nut_loc = False
                for spanner in current_usable_spanners_on_ground:
                    spanner_location = current_spanner_locations.get(spanner)
                    if spanner_location == nut_location:
                        usable_spanner_at_nut_loc = True
                        break # Found one at the nut location

                if usable_spanner_at_nut_loc:
                    cost_get_spanner_from_nut_loc = 1 # Pickup at nut location
                else:
                    # Find closest usable spanner from nut location
                    min_dist_from_nut_to_spanner = math.inf
                    for spanner in current_usable_spanners_on_ground:
                        spanner_location = current_spanner_locations.get(spanner)
                        if spanner_location is None: continue
                        dist_from_nut_to_spanner = self.get_distance(nut_location, spanner_location)
                        min_dist_from_nut_to_spanner = min(min_dist_from_nut_to_spanner, dist_from_nut_to_spanner)

                    if min_dist_from_nut_to_spanner != math.inf:
                        cost_get_spanner_from_nut_loc = min_dist_from_nut_to_spanner + 1 # Travel from nut + pickup

                cost_option_A = cost_to_reach_nut_loc + cost_get_spanner_from_nut_loc
                if cost_get_spanner_from_nut_loc == math.inf: cost_option_A = math.inf # If cannot get spanner from nut loc

                # Option B: Go get spanner first, then go to nut.
                min_cost_get_spanner_then_nut = math.inf
                for spanner in current_usable_spanners_on_ground:
                    spanner_location = current_spanner_locations.get(spanner)
                    if spanner_location is None: continue

                    cost_to_get_spanner = self.get_distance(man_location, spanner_location)
                    if cost_to_get_spanner == math.inf: continue

                    cost_from_spanner_to_nut = self.get_distance(spanner_location, nut_location)
                    if cost_from_spanner_to_nut == math.inf: continue

                    cost_path = cost_to_get_spanner + 1 + cost_from_spanner_to_nut
                    min_cost_get_spanner_then_nut = min(min_cost_get_spanner_then_nut, cost_path)

                cost_option_B = min_cost_get_spanner_then_nut

                cost_to_make_this_nut_ready = min(cost_option_A, cost_option_B)

            min_cost_one_nut_ready = min(min_cost_one_nut_ready, cost_to_make_this_nut_ready)

        # If min_cost_one_nut_ready is still infinity, it means no nut is reachable or no spanner is reachable from anywhere.
        if min_cost_one_nut_ready == math.inf:
             return math.inf

        # 8. Add the minimum cost to make one nut ready.
        cost += min_cost_one_nut_ready

        return cost
