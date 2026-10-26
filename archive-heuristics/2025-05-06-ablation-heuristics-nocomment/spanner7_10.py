# Assuming this import is available in the planner environment
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    return fact[1:-1].split()

# Helper function to match fact parts with patterns (not strictly needed for this heuristic but good practice)
# def match(fact, *args):
#     """Checks if the parts of a fact match the given patterns."""
#     parts = get_parts(fact)
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# class spannerHeuristic(Heuristic): # Use this line in the actual planner
class spannerHeuristic: # Use this for standalone testing/generation
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal state (all specified nuts tightened)
    by simulating a greedy sequence of actions for the single man agent.
    The heuristic calculates the cost for the man to sequentially address
    each untightened goal nut. For each nut, the man first ensures he is
    carrying a usable spanner (picking up the closest available one if needed),
    then walks to the nut's location, and finally performs the tighten action.
    The total heuristic is the sum of costs for these steps across all
    untightened goal nuts, processed in a greedy order (closest nut or spanner first).

    Assumptions:
    - There is exactly one man in the domain.
    - The problem is solvable (i.e., there are enough usable spanners initially
      to tighten all goal nuts, and the location graph is connected).
    - The goal only consists of (tightened ?n) predicates.
    - The names of object types 'man', 'nut', 'spanner', 'location' are implicitly known
      or can be inferred from predicate usage in the initial state. The current
      implementation infers the man object and assumes objects in 'usable'/'carrying'
      are spanners and objects in 'loose'/'tightened' are nuts.

    Heuristic Initialization:
    1. Parses static 'link' facts to build an adjacency list representation
       of the location graph (`self.location_graph`).
    2. Identifies all unique locations mentioned in the links or initial state
       and stores them in `self.all_locations`.
    3. Computes all-pairs shortest paths between all locations using BFS
       starting from each location and stores the distances in a dictionary `self.dist`,
       where `self.dist[l1][l2]` is the shortest distance from `l1` to `l2`.
    4. Identifies the set of nuts that need to be tightened from the task goals
       and stores their names in `self.goal_nuts`.
    5. Attempts to identify the man object by looking for the object involved
       in 'carrying' facts or the unique object 'at' a location that is not
       identified as a spanner or nut based on initial state predicates ('usable',
       'carrying', 'loose', 'tightened'). Stores the man's name in `self.man`.
       Includes a fallback to 'bob' if inference fails, based on example problems.

    Step-By-Step Thinking for Computing Heuristic (__call__):
    1. Access the current state (`node.state`) and convert it to a set (`state_facts`)
       for efficient lookups.
    2. Identify the man's current location by finding the fact `(at self.man ?l)`
       in `state_facts`.
    3. Determine if the man is currently carrying a *usable* spanner by checking
       for a fact `(carrying self.man ?s)` and verifying if `(usable ?s)` is also
       present in `state_facts`.
    4. Identify all usable spanners that are currently located at specific places
       (not carried by the man). This is done by finding objects `?s` for which
       `(usable ?s)` is in `state_facts` and `(at ?s ?l)` is in `state_facts`
       for some location `?l`. Store their locations in `usable_spanners_at_loc`.
    5. Identify the set of nuts that are currently `loose` in the state and are
       also present in the `self.goal_nuts` set. Store these nut objects in
       `loose_nuts`.
    6. Determine the subset of `loose_nuts` that are also in `self.goal_nuts`.
       Store these in `loose_goal_nuts`.
    7. If `loose_goal_nuts` is empty, all goal nuts are tightened, so return 0.
    8. Initialize the total heuristic cost `h` to 0.
    9. Initialize simulation variables: `current_man_location` starts at the man's
       actual location, `sim_carrying_usable` reflects the man's actual carrying
       status, `sim_available_spanner_locations` is a mutable copy of the identified
       available spanner locations, and `sim_loose_goal_nuts_objects` is a mutable
       copy of the `loose_goal_nuts` set. Also, create a map from loose goal nut
       objects to their current locations (`loose_goal_nut_locations_map`) by
       looking up their positions in the current state. If a loose goal nut is
       not found at any location, return `float('inf')` as the state is likely invalid
       or unsolvable.
    10. Enter a loop that continues as long as there are nut objects remaining in
        `sim_loose_goal_nuts_objects`.
    11. Inside the loop:
        a. If `sim_carrying_usable` is True:
           i. Find the location (`nut_loc`) of the nut object in
              `sim_loose_goal_nuts_objects` that has the minimum distance
              from `current_man_location` using the precomputed `self.dist`.
              Ensure the location is reachable (distance is not infinity).
           ii. If no reachable loose goal nut is found, return `float('inf')`.
           iii. Add this minimum distance to `h` (cost for walking).
           iv. Add 1 to `h` (cost for the `tighten_nut` action).
           v. Update `current_man_location` to `nut_loc`.
           vi. Remove the chosen nut object from `sim_loose_goal_nuts_objects`.
           vii. Set `sim_carrying_usable` to False, as the spanner is consumed.
        b. If `sim_carrying_usable` is False:
           i. Check if `sim_available_spanner_locations` is empty. If it is,
              and there are still nuts to tighten, the problem is unsolvable
              from this state with available spanners. Return `float('inf')`.
           ii. Find the location (`spanner_loc`) in `sim_available_spanner_locations`
               that has the minimum distance from `current_man_location`.
               Ensure the location is reachable (distance is not infinity).
           iii. If no reachable available spanner is found, return `float('inf')`.
           iv. Add this minimum distance to `h` (cost for walking).
           v. Add 1 to `h` (cost for the `pickup_spanner` action).
           vi. Update `current_man_location` to `spanner_loc`.
           vii. Remove `spanner_loc` from `sim_available_spanner_locations`.
           viii. Set `sim_carrying_usable` to True.
    12. Once the loop finishes (all loose goal nuts are processed), return the
        accumulated cost `h`.
    """
    def __init__(self, task):
        # Store goal nuts
        self.goal_nuts = {
            get_parts(goal)[1]
            for goal in task.goals
            if get_parts(goal)[0] == "tightened"
        }

        # Build location graph from static links and initial state locations
        self.location_graph = {}
        locations = set()

        # Add locations from link facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Add locations from initial state 'at' facts, just in case some locations exist but have no links
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 locations.add(parts[2])

        self.all_locations = list(locations) # Store all unique locations

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_node in self.all_locations:
            self.dist[start_node] = self._bfs(start_node)

        # Identify the man object (assuming there's only one man)
        self.man = None
        spanner_candidates = set()
        nut_candidates = set()
        man_candidates_at = set()

        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "usable" and len(parts) == 2:
                spanner_candidates.add(parts[1])
            elif parts[0] == "carrying" and len(parts) == 2:
                 # The second part is the spanner, the first is the carrier (man)
                 self.man = parts[1] # Found man via carrying fact
                 spanner_candidates.add(parts[2])
            elif parts[0] == "loose" and len(parts) == 2:
                nut_candidates.add(parts[1])
            elif parts[0] == "tightened" and len(parts) == 2:
                 nut_candidates.add(parts[1])
            elif parts[0] == "at" and len(parts) == 3:
                 obj_at_loc = parts[1]
                 man_candidates_at.add(obj_at_loc)

        if self.man is None:
             # Man was not found in a 'carrying' fact.
             # The man must be one of the objects 'at' a location
             # that is not a spanner or nut candidate.
             potential_men = man_candidates_at - spanner_candidates - nut_candidates
             if len(potential_men) == 1:
                 self.man = list(potential_men)[0]
             else:
                 # Fallback: Assume 'bob' based on examples if inference fails
                 # print("Warning: Could not uniquely identify the man object. Assuming 'bob'.")
                 self.man = 'bob' # Fallback


    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.all_locations}
        # Handle cases where start_node might not be in self.all_locations (e.g., malformed problem)
        if start_node not in self.all_locations:
             # This should not happen if self.all_locations is built correctly
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state
        state_facts = set(state)

        # 1. Identify relevant facts from the state
        man_location = None
        man_carrying_spanner_obj = None # The spanner object carried, or None
        usable_spanners_at_loc = {} # {spanner_obj: location_obj}
        loose_nuts = set() # {nut_obj}

        # First, find all usable spanner objects in the current state
        usable_spanner_objects_in_state = {get_parts(fact)[1] for fact in state_facts if get_parts(fact)[0] == "usable"}

        # Then, process other facts
        for fact in state_facts:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj == self.man:
                     man_location = loc
                 elif obj in usable_spanner_objects_in_state:
                     # This object is a usable spanner and is at a location
                     usable_spanners_at_loc[obj] = loc
             elif parts[0] == "carrying" and len(parts) == 2:
                 carrier, obj = parts[1], parts[2]
                 if carrier == self.man and obj in usable_spanner_objects_in_state:
                     # The man is carrying a spanner that is usable
                     man_carrying_spanner_obj = obj
             elif parts[0] == "loose" and len(parts) == 2:
                 nut = parts[1]
                 loose_nuts.add(nut)

        # Check if man_location was found (should always be true in valid states)
        if man_location is None:
             # print(f"Error: Man '{self.man}' location not found in state.")
             return float('inf') # Should not happen in a valid state

        # 6. Determine loose goal nuts
        loose_goal_nuts = {nut for nut in loose_nuts if nut in self.goal_nuts}

        # 7. If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # 8. Initialize heuristic cost and simulation state
        h = 0
        current_man_location = man_location
        sim_carrying_usable = man_carrying_spanner_obj is not None
        sim_available_spanner_locations = set(usable_spanners_at_loc.values()) # Use locations directly
        sim_loose_goal_nuts_objects = set(loose_goal_nuts) # Keep track of objects

        # Map loose goal nut object to its current location for easy lookup during simulation
        loose_goal_nut_locations_map = {}
        for nut_obj in sim_loose_goal_nuts_objects:
             # Find the location of this loose nut in the current state
             nut_loc = None
             for fact in state_facts:
                 parts = get_parts(fact)
                 if parts[0] == "at" and len(parts) == 3 and parts[1] == nut_obj:
                     nut_loc = parts[2]
                     break
             if nut_loc is None:
                 # This nut is loose and a goal, but not at any location? Unsolvable.
                 # print(f"Error: Loose goal nut {nut_obj} not found at any location in state.")
                 return float('inf') # Indicate unsolvable path
             loose_goal_nut_locations_map[nut_obj] = nut_loc


        # 10. Simulate greedy sequence
        while sim_loose_goal_nuts_objects:
            if sim_carrying_usable:
                # Man has a spanner, go to the closest loose goal nut
                closest_nut_loc = None
                closest_nut_obj = None
                min_dist_to_nut = float('inf')

                # Find the closest location among the remaining loose goal nuts
                for nut_obj in sim_loose_goal_nuts_objects:
                    nut_loc = loose_goal_nut_locations_map[nut_obj] # Get location from map
                    # Ensure the location is reachable from the current man location
                    if current_man_location in self.dist and nut_loc in self.dist[current_man_location]:
                         dist = self.dist[current_man_location][nut_loc]
                         if dist < min_dist_to_nut:
                             min_dist_to_nut = dist
                             closest_nut_loc = nut_loc
                             closest_nut_obj = nut_obj

                if closest_nut_loc is None or min_dist_to_nut == float('inf'):
                     # This means remaining goal nuts are at unreachable locations. Unsolvable.
                     # print("Error: Cannot find reachable loose goal nut location for simulation.")
                     return float('inf') # Indicate unsolvable path

                h += min_dist_to_nut  # Walk to nut
                h += 1  # Tighten nut
                current_man_location = closest_nut_loc
                sim_loose_goal_nuts_objects.remove(closest_nut_obj) # Remove the object
                sim_carrying_usable = False # Spanner is used up

            else: # Man needs a spanner
                # Go to the closest available usable spanner
                closest_spanner_loc = None
                min_dist_to_spanner = float('inf')

                for spanner_loc in sim_available_spanner_locations:
                    # Ensure the spanner location is reachable
                    if current_man_location in self.dist and spanner_loc in self.dist[current_man_location]:
                        dist = self.dist[current_man_location][spanner_loc]
                        if dist < min_dist_to_spanner:
                            min_dist_to_spanner = dist
                            closest_spanner_loc = spanner_loc

                if closest_spanner_loc is None or min_dist_to_spanner == float('inf'):
                    # No usable spanners left (or reachable), but nuts still need tightening. Unsolvable.
                    # print("Error: No usable spanners available (or reachable) to pick up for simulation.")
                    return float('inf') # Indicate unsolvable path

                h += min_dist_to_spanner  # Walk to spanner
                h += 1  # Pickup spanner
                current_man_location = closest_spanner_loc
                sim_available_spanner_locations.remove(closest_spanner_loc)
                sim_carrying_usable = True # Now carrying a usable spanner

        # 12. Return total cost
        return h
