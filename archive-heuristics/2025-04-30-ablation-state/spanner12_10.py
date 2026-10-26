from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts specified in the goal.
    It greedily calculates the cost for each loose goal nut by estimating the steps needed to:
    1. Acquire a usable spanner (if not already carrying one).
    2. Walk to the nut's location.
    3. Perform the tighten action.
    The cost is accumulated sequentially for each nut, assuming the man starts from his location after completing the previous nut's task.

    # Assumptions:
    - There is exactly one man in the problem.
    - Nuts specified in the goal are initially loose (the heuristic only considers currently loose goal nuts).
    - Spanners are consumed (become unusable) after one tighten action.
    - The man can only carry one spanner at a time.
    - All relevant locations (those mentioned in links, initial state, or goals) are included in the graph for distance calculation. Unreachable locations result in infinite cost.
    - There are enough usable spanners available initially to tighten all goal nuts in a solvable problem instance.

    # Heuristic Initialization
    - Parse static facts (`link`) to build the location graph.
    - Collect all relevant locations from static facts, initial state, and goals.
    - Compute all-pairs shortest path distances between these locations using BFS.
    - Parse object types to identify the man, spanners, and nuts.
    - Identify the set of nuts that need to be tightened based on the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location. If the man's location is unknown, return infinity.
    2. Identify which usable spanners are currently available (carried by the man or on the ground) and their locations.
    3. Identify the set of nuts that are currently loose and are specified in the goal. If this set is empty, the goal is reached, and the heuristic is 0.
    4. Sort the loose goal nuts based on their shortest distance from the man's current location in the current state. This determines the greedy order of tackling nuts.
    5. Initialize the total heuristic cost to 0. Keep track of the man's current location and whether he is carrying a usable spanner. Keep track of available usable spanners on the ground.
    6. Iterate through the sorted list of loose goal nuts:
        a. If the man is not currently carrying a usable spanner:
            i. Find the closest available usable spanner on the ground to the man's current location.
            ii. If no usable spanners are available on the ground or reachable, return infinity (as the task cannot be completed).
            iii. Add the walk distance to this spanner's location plus 1 (for the pickup action) to the total cost.
            iv. Update the man's current location to the spanner's location. Mark the man as carrying a usable spanner. Remove the picked-up spanner from the set of available ground spanners.
        b. Add the walk distance from the man's current location to the target nut's location to the total cost. If the nut location is unreachable, return infinity.
        c. Update the man's current location to the nut's location.
        d. Add 1 (for the tighten action) to the total cost. Mark the man as no longer carrying a usable spanner (as it's consumed).
    7. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, object types, and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        self.objects = task.objects # Access task objects
        self.initial_state = task.initial_state # Access initial state facts

        # Parse object types
        self.object_types = {} # {obj_name: type_name}
        self.types_to_objects = {} # {type_name: {obj_name}}
        for obj_str in self.objects:
            parts = obj_str.split(' - ')
            if len(parts) == 2:
                obj_name, type_name = parts[0], parts[1]
                self.object_types[obj_name] = type_name
                self.types_to_objects.setdefault(type_name, set()).add(obj_name)

        # Identify the man object name (assuming there's only one man)
        self.man_name = next(iter(self.types_to_objects.get('man', set())), None)
        if self.man_name is None:
             # This should not happen in valid spanner problems
             print("Warning: Could not identify the man object.")

        # Collect all relevant locations from static facts, initial state, and goals
        locations = set()
        self.location_graph = {} # Initialize graph here

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                # Build location graph simultaneously
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional

        # Add locations mentioned in initial state and goals
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                  # The third part is the location
                  locations.add(parts[2])

        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at' and len(parts) == 3:
                  # The third part is the location
                  locations.add(parts[2])

        self.locations = list(locations) # Store all known locations

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = self._bfs(start_node)

        # Store goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])


    def _bfs(self, start_node):
        """Computes shortest path distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.locations}

        if start_node not in self.locations:
             # Start node is not in the collected locations.
             # Only distance to itself is 0 if it was a valid location name.
             # For safety, if it's a known location name, set dist to itself as 0.
             if start_node in distances:
                 distances[start_node] = 0
             return distances # All others remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph (handle isolated nodes)
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def get_dist(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        # If loc1 or loc2 were not collected during init, they are unknown or isolated.
        # Distance is infinity unless they are the same location.
        if loc1 == loc2: return 0
        if loc1 not in self.dist or loc2 not in self.dist[loc1]:
             return float('inf')
        return self.dist[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify man and his location
        man_loc = None
        if self.man_name:
            for fact in state:
                 parts = get_parts(fact)
                 if parts[0] == 'at' and parts[1] == self.man_name:
                      man_loc = parts[2]
                      break
        if man_loc is None:
             # Man is not located anywhere? Should not happen in valid states.
             return float('inf') # Indicate unsolvable state

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts = {} # {nut_name: location}
        current_loose_nuts = set()
        nut_locations_in_state = {} # {nut_name: location}

        # Get all nut objects identified during init
        all_nuts = self.types_to_objects.get('nut', set())

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'loose' and parts[1] in all_nuts:
                current_loose_nuts.add(parts[1])
            elif parts[0] == 'at' and parts[1] in all_nuts:
                 nut_name, loc_name = parts[1], parts[2]
                 nut_locations_in_state[nut_name] = loc_name

        for nut_name in self.goal_nuts:
             # Only consider goal nuts that are currently loose and have a location in the state
             if nut_name in current_loose_nuts and nut_name in nut_locations_in_state:
                  loose_goal_nuts[nut_name] = nut_locations_in_state[nut_name]
             # If a goal nut is not in state as loose and at a location, it's assumed tightened or irrelevant.

        # If no loose goal nuts, we are at the goal
        if not loose_goal_nuts:
            return 0

        # 3. Identify usable spanners (carried and on ground) and their locations
        usable_spanners_carried = None # Store the spanner name if carried
        usable_spanners_ground = {} # {spanner_name: location}
        current_usable_spanners = set()
        spanner_locations_in_state = {} # {spanner_name: location}

        # Get all spanner objects identified during init
        all_spanners = self.types_to_objects.get('spanner', set())

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'usable' and parts[1] in all_spanners:
                current_usable_spanners.add(parts[1])
            elif parts[0] == 'at' and parts[1] in all_spanners:
                 spanner_name, loc_name = parts[1], parts[2]
                 spanner_locations_in_state[spanner_name] = loc_name


        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and parts[1] == self.man_name:
                 spanner_name = parts[2]
                 if spanner_name in current_usable_spanners:
                      usable_spanners_carried = spanner_name

        for spanner_name in current_usable_spanners:
             # Only consider usable spanners that have a location in the state (on ground)
             if spanner_name in spanner_locations_in_state:
                  usable_spanners_ground[spanner_name] = spanner_locations_in_state[spanner_name]


        # 4. Sort loose goal nuts by distance from man's current location
        nuts_to_tighten = list(loose_goal_nuts.keys())
        # Sort by distance from man_loc to nut_loc
        # Handle cases where nut_loc might not be in the graph (get_dist handles this)
        nuts_to_tighten.sort(key=lambda nut: self.get_dist(man_loc, loose_goal_nuts[nut]))

        # 5. Calculate heuristic cost greedily
        h = 0
        current_man_loc = man_loc
        man_has_usable_spanner = usable_spanners_carried is not None
        available_ground_spanners = set(usable_spanners_ground.keys()) # Use a mutable set

        for nut_name in nuts_to_tighten:
            target_nut_loc = loose_goal_nuts[nut_name]

            # Cost to get a usable spanner if needed
            if not man_has_usable_spanner:
                # Find closest available ground spanner to current_man_loc
                closest_spanner = None
                min_dist_to_spanner = float('inf')
                spanner_loc = None

                # Check if a usable spanner is already at the current man location (could be the nut location)
                spanners_at_current_loc = [
                    s for s in available_ground_spanners
                    if usable_spanners_ground.get(s) == current_man_loc
                ]

                if spanners_at_current_loc:
                     # Pick up one spanner at the current location
                     closest_spanner = spanners_at_current_loc[0] # Just pick the first one
                     spanner_loc = current_man_loc
                     min_dist_to_spanner = 0
                else:
                    # Find closest spanner elsewhere
                    for s in available_ground_spanners:
                        loc_s = usable_spanners_ground.get(s)
                        if loc_s is not None: # Ensure spanner is on the ground and has a location
                            d = self.get_dist(current_man_loc, loc_s)
                            if d == float('inf'): continue # Skip unreachable spanners
                            if d < min_dist_to_spanner:
                                min_dist_to_spanner = d
                                closest_spanner = s
                                spanner_loc = loc_s

                if closest_spanner is None:
                     # No usable spanners left on the ground or reachable.
                     # If there are still nuts to tighten, this state is likely unsolvable.
                     return float('inf')

                # Walk to spanner and pick it up
                h += min_dist_to_spanner + 1 # walk + pickup
                current_man_loc = spanner_loc # Man is now at spanner location
                man_has_usable_spanner = True
                available_ground_spanners.remove(closest_spanner)

            # Cost to get man to target_nut_loc (man is currently at current_man_loc, carrying spanner)
            dist_to_nut = self.get_dist(current_man_loc, target_nut_loc)
            if dist_to_nut == float('inf'):
                 # Cannot reach the nut location from the current location
                 return float('inf')
            h += dist_to_nut
            current_man_loc = target_nut_loc

            # Tighten the nut
            h += 1 # tighten action
            man_has_usable_spanner = False # Spanner used becomes unusable

        return h
