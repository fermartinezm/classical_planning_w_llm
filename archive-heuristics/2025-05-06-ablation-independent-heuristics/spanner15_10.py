import math
from collections import deque
from heuristics.heuristic_base import Heuristic
# Assuming Task class is available in the environment where this heuristic runs
# from task import Task # Not strictly needed to import if Task object is just passed in


# Helper function to parse facts like '(predicate obj1 obj2)'
def parse_fact(fact_str):
    """Parses a PDDL fact string into predicate and objects."""
    # Remove parentheses and split by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

# Helper function to get object location from state
def get_object_location(state, obj_name):
    """Finds the current location of an object in the state."""
    for fact_str in state:
        if fact_str.startswith('(at '):
            predicate, objects = parse_fact(fact_str)
            if objects[0] == obj_name:
                return objects[1]
    return None # Object not found at any location (e.g., carried)

# Helper function to get the spanner the man is carrying
def get_carried_spanner(state, man_name):
    """Returns the name of the spanner the man is carrying, or None."""
    for fact_str in state:
        if fact_str.startswith('(carrying '):
            predicate, objects = parse_fact(fact_str)
            if objects[0] == man_name:
                return objects[1] # Return the spanner name
    return None # Not carrying any spanner

# Helper function to check if a spanner is usable
def is_spanner_usable(state, spanner_name):
    """Checks if a spanner is usable in the current state."""
    return f'(usable {spanner_name})' in state

# Helper function to check if a nut is loose
def is_nut_loose(state, nut_name):
    """Checks if a nut is loose in the current state."""
    return f'(loose {nut_name})' in state


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all required nuts tightened)
    by simulating a greedy strategy. The strategy involves repeatedly finding the
    closest required task (either picking up a usable spanner if one is needed,
    or tightening the closest loose nut if a usable spanner is carried) and adding
    the estimated travel cost and action cost (1 for pickup, 1 for tighten) to the
    heuristic value. This process continues until all loose nuts are considered
    tightened in the simulation.

    Assumptions:
    - There is exactly one man object in the domain.
    - Nuts remain at their initial locations throughout the plan.
    - Spanners are either at their initial location (if not carried and usable) or carried.
      (The domain does not have a drop action, and tighten removes (usable ?s) but not (at ?s ?l)).
    - The location graph defined by 'link' predicates is undirected.
    - The problem is solvable if the heuristic is finite.

    Heuristic Initialization:
    The constructor performs the following steps:
    1. Identifies all objects (man, spanners, nuts, locations) by parsing facts
       from the initial state, goals, static information, and operator definitions.
       It assumes a single man object.
    2. Stores the initial locations of all locatable objects (man, spanners, nuts)
       from the initial state. Nuts' locations are considered static.
    3. Builds an adjacency list representation of the location graph based on
       'link' predicates in the static information. All identified locations are
       included, even if isolated.
    4. Computes all-pairs shortest path distances between all locations using
       Breadth-First Search (BFS) starting from each location. These distances
       represent the minimum number of 'walk' actions required to travel between
       any two locations.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic `h(state)` is computed as follows:
    1. Identify all nuts that are currently loose in the state and are part of the goal.
       If there are no such nuts, the heuristic value is 0.
    2. Identify all spanners that are currently usable in the state and are located
       at a specific location (i.e., not currently carried by the man). These are
       the available usable spanners for pickup.
    3. Determine if the man is currently carrying a usable spanner.
    4. Get the man's current location from the state.
    5. Initialize the estimated cost `h = 0`.
    6. Create working lists/sets for the nuts that still need to be tightened
       (initialized with the loose goal nuts from step 1) and the usable spanners
       available for pickup (initialized from step 2).
    7. Set the man's current location for the simulation (`sim_current_loc`) to
       his actual current location (from step 4).
    8. Set the man's spanner status for the simulation (`sim_carrying_usable`)
       based on whether he is currently carrying a usable spanner (from step 3).
    9. Enter a loop that continues as long as there are nuts remaining to be tightened
       in the simulation list:
        a. If `sim_carrying_usable` is True (the man has a usable spanner in the simulation):
           i. Find the nut `n` from the `remaining_loose_nuts` list whose location
              `NutLoc(n)` is closest to `sim_current_loc` according to the precomputed
              shortest path distances.
           ii. If no reachable loose nut is found (distance is infinity), the problem
               is unsolvable from this state, return `float('inf')`.
           iii. Add the shortest distance (`travel_cost`) to `h`.
           iv. Update `sim_current_loc` to `NutLoc(n)`.
           v. Add 1 to `h` for the `tighten_nut` action.
           vi. Set `sim_carrying_usable` to False, as the spanner is considered used.
           vii. Remove the chosen nut `n` from the `remaining_loose_nuts` list.
        b. If `sim_carrying_usable` is False (the man needs a usable spanner in the simulation):
           i. Find the spanner `s` from the `available_usable_spanners_for_pickup` list
              whose current location `SpannerLoc(s)` is closest to `sim_current_loc`.
           ii. If no available usable spanner is found (list is empty or all are unreachable),
               the problem is unsolvable from this state, return `float('inf')`.
           iii. Add the shortest distance (`travel_cost`) to `h`.
           iv. Update `sim_current_loc` to `SpannerLoc(s)`.
           v. Add 1 to `h` for the `pickup_spanner` action.
           vi. Set `sim_carrying_usable` to True, as a spanner is acquired.
           vii. Remove the chosen spanner `s` from the
               `available_usable_spanners_for_pickup` list.
    10. Once the loop finishes (all loose goal nuts have been considered tightened in the simulation),
        return the accumulated heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

        # 1. Identify objects and their types
        self.men = set()
        self.spanners = set()
        self.nuts = set()
        self.locations = set()
        self.locatables = set()

        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for op in task.operators:
            all_facts |= set(op.preconditions) | set(op.add_effects) | set(op.del_effects)

        for fact_str in all_facts:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'carrying':
                self.men.add(objects[0])
                self.spanners.add(objects[1])
                self.locatables.add(objects[0])
                self.locatables.add(objects[1])
            elif predicate == 'usable':
                self.spanners.add(objects[0])
                self.locatables.add(objects[0])
            elif predicate == 'tightened' or predicate == 'loose':
                self.nuts.add(objects[0])
                self.locatables.add(objects[0])
            elif predicate == 'link':
                self.locations.add(objects[0])
                self.locations.add(objects[1])
            elif predicate == 'at':
                self.locatables.add(objects[0])
                self.locations.add(objects[1])

        # Assume there is exactly one man
        if len(self.men) != 1:
             # This heuristic assumes a single man. Handle error or pick one.
             # Picking the first one found.
             if not self.men:
                  # This case should ideally be caught by a PDDL parser earlier
                  # but handling defensively.
                  self.man_name = None # Indicate no man found
             else:
                  self.man_name = list(self.men)[0]
        else:
             self.man_name = list(self.men)[0]

        # If no man is found, the heuristic cannot be computed.
        if self.man_name is None:
             # This heuristic is unusable without a man object.
             # We could raise an error or return inf for any state.
             # Let's make __call__ handle this by returning inf.
             pass # Continue init, __call__ will check self.man_name


        # 2. Store initial locations (especially for nuts, which don't move)
        self.initial_locations = {}
        for fact_str in task.initial_state:
            if fact_str.startswith('(at '):
                predicate, objects = parse_fact(fact_str)
                self.initial_locations[objects[0]] = objects[1]

        # 3. Build location graph
        self.location_graph = {loc: set() for loc in self.locations}
        for fact_str in task.static:
            if fact_str.startswith('(link '):
                predicate, objects = parse_fact(fact_str)
                l1, l2 = objects[0], objects[1]
                # Ensure locations are in our identified set before adding link
                if l1 in self.locations and l2 in self.locations:
                    self.location_graph[l1].add(l2)
                    self.location_graph[l2].add(l1)

        # 4. Compute all-pairs shortest paths
        self.distances = {}
        # Ensure all identified locations are keys in distances, even if isolated
        for loc in self.locations:
             self.distances[loc] = self._bfs(loc, self.location_graph)


    def _bfs(self, start_node, graph):
        """Performs BFS to find distances from start_node to all other nodes in the graph."""
        distances = {node: math.inf for node in self.locations} # Use all identified locations
        if start_node in distances: # Ensure start_node is a known location
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                # Only process if the node is in the graph (has links)
                if current_node in graph:
                    for neighbor in graph[current_node]:
                        if distances[neighbor] == math.inf:
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)
        # If start_node was not in self.locations, distances will be all inf, which is correct.
        return distances


    def __call__(self, node):
        """Computes the domain-dependent heuristic for the given state."""
        state = node.state

        # If no man object was found during initialization, heuristic is infinite.
        if self.man_name is None:
             return math.inf

        # 1. Identify loose nuts in the current state that are part of the goal
        # The goal is a set of (tightened nutX) facts. We need to tighten nuts
        # that are currently loose AND are in the goal.
        goal_nuts = {parse_fact(g)[1][0] for g in self.task.goals if g.startswith('(tightened ')}
        loose_nuts_in_state = {n for n in self.nuts if is_nut_loose(state, n)}
        loose_goal_nuts = list(loose_nuts_in_state.intersection(goal_nuts))

        # If no loose goal nuts, the goal is reached from this perspective
        if not loose_goal_nuts:
            return 0

        # 2. Identify usable spanners available for pickup
        # These are usable spanners that are currently at a location and not carried by the man
        available_usable_spanners_for_pickup = []
        for spanner in self.spanners:
            if is_spanner_usable(state, spanner):
                spanner_loc = get_object_location(state, spanner)
                # Spanner must be at a location to be picked up
                if spanner_loc is not None:
                     # Check if the man is carrying this specific spanner in the current state
                     carried_spanner_name = get_carried_spanner(state, self.man_name)
                     if carried_spanner_name != spanner:
                         available_usable_spanners_for_pickup.append(spanner)

        # 3. Determine if the man is currently carrying a usable spanner
        man_carrying_usable_in_state = False
        carried_spanner_name = get_carried_spanner(state, self.man_name)
        if carried_spanner_name is not None and is_spanner_usable(state, carried_spanner_name):
             man_carrying_usable_in_state = True

        # 4. Get the man's current location
        man_current_loc = get_object_location(state, self.man_name)
        if man_current_loc is None:
             # Man is not at a location? This state is likely invalid/unreachable.
             return math.inf
        # Ensure man_current_loc is a known location in our distance map
        if man_current_loc not in self.locations:
             # Man is at an unknown location? Unreachable.
             return math.inf


        # 5. Initialize simulation state
        h = 0
        sim_current_loc = man_current_loc
        sim_carrying_usable = man_carrying_usable_in_state
        sim_loose_nuts = list(loose_goal_nuts) # Create a mutable copy
        sim_available_usable_spanners = list(available_usable_spanners_for_pickup) # Create a mutable copy

        # 6. Simulate the greedy process
        while sim_loose_nuts:
            if sim_carrying_usable:
                # Man has a spanner, go tighten the closest loose nut
                closest_nut = None
                min_dist = math.inf
                target_loc = None

                for nut in sim_loose_nuts:
                    # Nuts are assumed to be at their initial locations
                    nut_loc = self.initial_locations.get(nut)
                    if nut_loc is None or nut_loc not in self.locations:
                         # Nut location unknown or not a valid location
                         continue # Skip this nut, maybe it makes the problem unsolvable

                    dist = self.distances.get(sim_current_loc, {}).get(nut_loc, math.inf)
                    if dist < min_dist:
                        min_dist = dist
                        closest_nut = nut
                        target_loc = nut_loc

                if closest_nut is None or min_dist == math.inf:
                    # Cannot reach any remaining loose goal nut
                    return math.inf

                # Add travel cost
                h += min_dist
                # Update simulation location
                sim_current_loc = target_loc
                # Add tighten action cost
                h += 1
                # Spanner is used
                sim_carrying_usable = False
                # Nut is tightened
                sim_loose_nuts.remove(closest_nut)

            else:
                # Man needs a spanner, go pick up the closest available usable one
                closest_spanner = None
                min_dist = math.inf
                target_loc = None

                # Need to iterate through available spanners to find the closest one
                for spanner in sim_available_usable_spanners:
                    # Get the current location of the spanner from the *actual* state
                    spanner_loc = get_object_location(state, spanner)
                    if spanner_loc is None or spanner_loc not in self.locations:
                         # Spanner is not at a known location (e.g., carried by someone else, or initial state issue)
                         continue # Skip this spanner

                    dist = self.distances.get(sim_current_loc, {}).get(spanner_loc, math.inf)
                    if dist < min_dist:
                        min_dist = dist
                        closest_spanner = spanner
                        target_loc = spanner_loc

                if closest_spanner is None or min_dist == math.inf:
                    # Cannot reach any available usable spanner
                    return math.inf

                # Add travel cost
                h += min_dist
                # Update simulation location
                sim_current_loc = target_loc
                # Add pickup action cost
                h += 1
                # Man now has a usable spanner
                sim_carrying_usable = True
                # Spanner is picked up (remove from available list for simulation)
                sim_available_usable_spanners.remove(closest_spanner)

        # All loose goal nuts considered tightened in simulation
        return h
