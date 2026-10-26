from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Placeholder for the base class if not provided in the execution environment
# This allows the code to be tested standalone if needed, but the final output
# should assume the base class is imported from the specified path.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            self.initial_state = task.initial_state

        def __call__(self, node):
            raise NotImplementedError


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
    # Use zip with min length to handle cases where fact might have more parts than args (e.g., typed objects)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It considers the travel cost for the man to reach nut locations and spanner locations,
    as well as the cost of picking up spanners and tightening nuts. It greedily
    processes loose goal nuts, simulating the man's path to acquire a usable spanner
    and then travel to the nut's location for tightening.

    # Assumptions
    - Each goal nut requires one usable spanner.
    - A spanner becomes unusable after one use for tightening a nut.
    - The man can carry at most one spanner at a time.
    - The man must pick up a usable spanner before tightening a nut.
    - The graph of locations connected by 'link' predicates is connected, or at least
      all relevant locations (man start, spanner locations, nut locations) are in the
      same connected component.
    - The problem is solvable (enough usable spanners exist).
    - There is exactly one object of type 'man'.

    # Heuristic Initialization
    - Identify the name of the man object.
    - Build a graph of locations based on 'link' predicates and object locations found in initial/goal states.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Identify the set of nuts that are goals (need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic calculates the cost by simulating a greedy sequence of actions
    for the man to tighten all loose goal nuts:

    1.  Identify the man's current location from the state.
    2.  Identify all loose nuts that are goal conditions and their locations from the state.
    3.  Identify all currently usable spanners and their locations from the state.
    4.  Check if the man is currently carrying a usable spanner based on the state.
        If he is, this spanner is considered available in his hand for the first nut task.
    5.  Initialize total cost to 0.
    6.  Initialize the man's current location for the simulation.
    7.  Initialize the set of available usable spanners *on the ground* for the simulation.
    8.  Initialize a flag indicating if the man is currently carrying a usable spanner
        in the simulation based on step 4.
    9.  Iterate through the loose goal nuts. A greedy approach is used: process nuts
        in increasing order of the initial distance from the man's starting location.
        For each nut:
        a.  Calculate the cost for the man to walk from his current simulated location
            to the nut's location. Add this cost to the total. Update the man's
            simulated location to the nut's location.
        b.  If the man is not currently carrying a usable spanner in the simulation:
            i.  Find the closest available usable spanner *on the ground* (minimizing walk distance
                from the man's current simulated location to the spanner's location).
            ii. If no usable spanner is found, the state is likely unsolvable for this nut;
                return infinity.
            iii. Calculate the cost for the man to walk to this spanner's location.
                 Add this cost to the total. Update the man's simulated location.
            iv. Add 1 for the 'pickup_spanner' action.
            v.  Mark the man as carrying a usable spanner in the simulation. Remove
                the used spanner from the set of available usable spanners *on the ground*.
            vi. Calculate the cost for the man to walk back from the spanner's location
                to the nut's location (if different). Add this cost to the total.
                Update the man's simulated location.
        c.  Add 1 for the 'tighten_nut' action.
        d.  Mark the man as no longer carrying a usable spanner in the simulation
            (as it's now unusable).
    10. The total accumulated cost is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        identifying the man object, and precomputing location distances.
        """
        super().__init__(task)

        # Identify the man object name (assuming there's only one man)
        self.man_obj = None
        # Look for an object involved in a 'carrying' predicate or an 'at' predicate
        # where the object name looks like a man name ('bob' or starts with 'man').
        # Prioritize 'carrying' as it uniquely identifies the man object.
        for fact in self.initial_state:
             if match(fact, "carrying", "*", "*"):
                 # The first argument of 'carrying' is the man object
                 self.man_obj = get_parts(fact)[1]
                 break
        # If not found via 'carrying', look for 'at'
        if self.man_obj is None:
             for fact in self.initial_state:
                 if match(fact, "at", "*", "*"):
                     obj, loc = get_parts(fact)[1:]
                     # Heuristic guess: object starting with 'man' or named 'bob' is the man
                     if obj.lower().startswith('man') or obj.lower() == 'bob':
                         self.man_obj = obj
                         break

        if self.man_obj is None:
             # If man object still not found, print warning. Heuristic might fail later.
             print("Warning: Could not identify the man object name in initial state.")
             # The heuristic will likely return infinity if man_location cannot be found in __call__


        # Identify all locations from static facts (links) and initial/goal state ('at' facts)
        locations = set()
        self.location_graph = {} # Adjacency list

        # Extract locations and build graph from link facts
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Add locations from initial and goal states ('at' facts)
        all_relevant_facts = set(task.initial_state) | set(task.goals)
        for fact in all_relevant_facts:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) >= 3: # Ensure it has at least predicate, obj, loc
                    loc = parts[2] # Location is the third part
                    locations.add(loc)
                    self.location_graph.setdefault(loc, []) # Ensure all locations are keys

        self.locations = list(locations) # Store as list

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not in our known locations graph, cannot reach anything
             return distances # All remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.location_graph: # Check if node exists in graph keys
                for neighbor in self.location_graph.get(current_node, []): # Use .get for safety
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Get the precomputed distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # Handle cases where locations might not be in the precomputed graph
             # (e.g., isolated locations not linked or mentioned in initial/goal 'at' facts)
             # If they are the same location, distance is 0.
             if loc1 == loc2:
                 return 0
             # If they are different and not reachable (or one/both unknown), return infinity
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Identify man's current location
        man_location = None
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    man_location = get_parts(fact)[2]
                    break

        if man_location is None:
             # Man location not found in state, problem state is invalid or unsolvable
             return float('inf')


        # 2. Identify loose nuts that are goals and their locations
        loose_goal_nuts_info = [] # List of (nut_name, location)
        # We need nut locations from the current state
        current_nut_locations = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Assuming objects that are nuts are in the goal_nuts set
                 if obj in self.goal_nuts:
                     current_nut_locations[obj] = loc

        for nut in self.goal_nuts:
            is_loose = False
            for fact in state:
                if match(fact, "loose", nut):
                    is_loose = True
                    break # Found loose predicate

            if is_loose and nut in current_nut_locations:
                 loose_goal_nuts_info.append((nut, current_nut_locations[nut]))


        # If no loose goal nuts, goal is reached (should have been caught earlier, but safety)
        if not loose_goal_nuts_info:
             return 0

        # 3. Identify currently usable spanners and their locations (only those on the ground)
        available_usable_spanners_on_ground_info = [] # List of (spanner_name, location)
        carried_spanner_obj = None
        man_currently_carrying_usable = False

        # Find carried spanner first
        if self.man_obj:
            for fact in state:
                 if match(fact, "carrying", self.man_obj, "*"):
                     carried_spanner_obj = get_parts(fact)[2]
                     # Check if the carried spanner is usable
                     for usable_fact in state:
                         if match(usable_fact, "usable", carried_spanner_obj):
                             man_currently_carrying_usable = True
                             break # Found usable carried spanner
                     break # Found if man is carrying anything

        # Find usable spanners on the ground
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # If this usable spanner is the one being carried, it's not on the ground
                if spanner == carried_spanner_obj:
                    continue

                # Find location if it's on the ground
                spanner_location = None
                for loc_fact in state:
                    if match(loc_fact, "at", spanner, "*"):
                        spanner_location = get_parts(loc_fact)[2]
                        break

                if spanner_location:
                    available_usable_spanners_on_ground_info.append((spanner, spanner_location))


        # 5. Initialize simulation state
        total_cost = 0
        current_sim_man_location = man_location
        # man_currently_carrying_usable is already initialized based on state
        # available_usable_spanners_on_ground_info is already initialized based on state

        # 6. Sort loose goal nuts by initial distance from man (greedy processing order)
        # Use the precomputed distance from the *initial* man location for sorting the list once.
        # Need to handle cases where a nut location might not be in the precomputed graph
        # (e.g., isolated location). Sort unreachable nuts last.
        loose_goal_nuts_info.sort(key=lambda item: self.get_distance(man_location, item[1]))


        # 7. Iterate through loose goal nuts
        for nut, nut_location in loose_goal_nuts_info:
            # Check if nut location is reachable from current simulated location
            dist_to_nut = self.get_distance(current_sim_man_location, nut_location)
            if dist_to_nut == float('inf'):
                 # Nut is unreachable from current simulated location
                 return float('inf')

            # a. Cost to get man to nut location from current simulated location
            total_cost += dist_to_nut
            current_sim_man_location = nut_location # Man is now at nut location

            # b. Cost to get a usable spanner if needed
            if not man_currently_carrying_usable:
                # Find the closest available usable spanner on the ground
                closest_spanner_info = None
                min_dist_to_spanner = float('inf')

                # Prioritize spanners already at the current location (nut_location)
                spanners_at_current_loc = [(s, l) for (s, l) in available_usable_spanners_on_ground_info if l == current_sim_man_location]
                if spanners_at_current_loc:
                    # If spanners are at the current location, picking one up costs 1 + 0 walk
                    # Just take the first one found at this location
                    closest_spanner_info = spanners_at_current_loc[0]
                    min_dist_to_spanner = 0 # Walk cost is 0
                else:
                    # No spanners at current location, look elsewhere on the ground
                    for spanner, spanner_loc in available_usable_spanners_on_ground_info:
                        dist_to_spanner = self.get_distance(current_sim_man_location, spanner_loc)
                        if dist_to_spanner < min_dist_to_spanner:
                            min_dist_to_spanner = dist_to_spanner
                            closest_spanner_info = (spanner, spanner_loc)

                if closest_spanner_info is None or min_dist_to_spanner == float('inf'):
                    # Needed spanner but none reachable on the ground
                    return float('inf')

                spanner_to_pickup, spanner_loc = closest_spanner_info

                # Walk to spanner location (if not already there)
                if min_dist_to_spanner > 0:
                    total_cost += min_dist_to_spanner
                    current_sim_man_location = spanner_loc # Man is now at spanner location

                # Pickup spanner
                total_cost += 1
                man_currently_carrying_usable = True
                # Remove the used spanner from the available list on the ground
                available_usable_spanners_on_ground_info = [
                    (s, l) for (s, l) in available_usable_spanners_on_ground_info if s != spanner_to_pickup
                ]

                # Walk back to nut location (if spanner wasn't at nut location)
                if current_sim_man_location != nut_location:
                    cost_walk_back_to_nut = self.get_distance(current_sim_man_location, nut_location)
                    if cost_walk_back_to_nut == float('inf'):
                         # Should not happen if nut location was reachable initially
                         return float('inf')
                    total_cost += cost_walk_back_to_nut
                    current_sim_man_location = nut_location # Man is now back at nut location

            # c. Tighten nut
            total_cost += 1
            man_currently_carrying_usable = False # Spanner used, no longer usable/carried in usable state

        return total_cost
