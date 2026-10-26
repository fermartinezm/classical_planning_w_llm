import math
from collections import deque

# Assume the Task and Operator classes are available from code-file-task.py

def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding brackets and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

def bfs(start_node, graph):
    """
    Performs BFS from start_node on the graph to find shortest distances.
    Returns a dictionary {node: distance}.
    """
    distances = {node: math.inf for node in graph}
    if start_node in graph: # Ensure start_node is in the graph keys
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph: # Should always be true if queue elements are from graph keys
                for neighbor in graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    # If start_node was not in graph, all distances remain inf, which is correct.
    return distances

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal state (all goal nuts
    tightened) by summing the number of remaining loose goal nuts and
    an estimated travel/pickup cost to enable the next useful action.
    The next useful action is either tightening a nut (if preconditions met),
    moving to the nearest loose nut location (if carrying a spanner), or
    moving to the nearest usable spanner location and picking it up (if not
    carrying a spanner). Shortest path distances between locations are
    precomputed using BFS on the graph defined by the 'link' predicates.

    Assumptions:
    - There is exactly one man object in the domain.
    - 'link' predicates represent bidirectional connections between locations.
    - Solvable problems always have enough usable spanners initially.
    - The graph of locations defined by 'link' predicates is connected, or
      all locations relevant to the problem (man start, nut locations,
      spanner locations) are in the same connected component.

    Heuristic Initialization:
    The constructor builds an undirected graph of locations based on the
    static 'link' facts. It then collects all unique location names mentioned
    in static facts and initial state 'at' facts. It computes the shortest
    path distance between every pair of these collected locations using BFS
    starting from each location. It also identifies the name of the man object
    and the set of goal nuts from the task definition.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of nuts that are goals and are currently loose
       in the given state. Let N_loose_in_goal be the count.
    2. If N_loose_in_goal is 0, the state is a goal state, return 0.
    3. The base heuristic value is N_loose_in_goal (representing the minimum
       number of tighten actions required).
    4. Find the current location of the man.
    5. Check if the man is currently carrying a usable spanner.
    6. Find the locations of all currently loose goal nuts.
    7. Find the locations of all currently usable spanners that are at a location
       (not being carried).
    8. Calculate the cost to reach the "next action zone":
       - If the man is carrying a usable spanner: The next useful step is
         to go to the nearest location with a loose goal nut. Add the shortest
         distance from the man's current location to the nearest loose goal
         nut location to the heuristic. If there are loose goal nuts but
         no locations found (e.g., nut location not in graph or nut not at a location),
         this implies unsolvability or a problem definition issue; return infinity.
       - If the man is NOT carrying a usable spanner: The next useful step is
         to go to the nearest location with a usable spanner and pick it up.
         Add the shortest distance from the man's current location to the
         nearest usable spanner location plus 1 (for the pickup action) to
         the heuristic. If there are loose goal nuts but no usable spanners
         are available at any location, return infinity.
    9. Return the calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by building the location graph and
        precomputing shortest path distances.
        """
        self.task = task
        self.graph = {}
        self.locations = set()
        self.goal_nuts = set()
        self.man_name = None

        # Build graph from static links and collect all locations
        for fact_string in task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'link':
                loc1, loc2 = fact[1], fact[2]
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Assume bidirectional links
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Collect locations mentioned in initial state 'at' facts
        # This ensures locations where objects start are included, even if not linked
        for fact_string in task.initial_state:
             fact = parse_fact(fact_string)
             if fact[0] == 'at':
                 # fact is (at obj loc)
                 self.locations.add(fact[2])

        # Ensure all collected locations are keys in the graph dictionary, even if they have no links
        for loc in self.locations:
             self.graph.setdefault(loc, [])


        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = bfs(start_node, self.graph)

        # Identify goal nuts
        for goal_fact_string in task.goals:
            goal_fact = parse_fact(goal_fact_string)
            if goal_fact[0] == 'tightened':
                self.goal_nuts.add(goal_fact[1])

        # Identify the man object name
        # Look for the object involved in a 'carrying' fact in the initial state
        # If not found, look for the unique object that is 'at' a location but is not a spanner or nut
        # (assuming spanners/nuts are identified by being arguments to usable/loose/tightened)
        locatable_objects_at_start = set()
        spanner_like_objects = set() # Objects that appear in spanner/nut predicates
        nut_like_objects = set()

        for fact_string in task.initial_state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at':
                obj = fact[1]
                locatable_objects_at_start.add(obj)
            elif fact[0] == 'carrying':
                 self.man_name = fact[1] # Found man via carrying fact
            elif fact[0] == 'usable':
                 spanner_like_objects.add(fact[1])
            elif fact[0] == 'loose' or fact[0] == 'tightened':
                 nut_like_objects.add(fact[1])

        if self.man_name is None:
             # Man not found via carrying, find the object at a location that isn't a spanner or nut
             potential_men = locatable_objects_at_start - spanner_like_objects - nut_like_objects
             if len(potential_men) == 1:
                 self.man_name = potential_men.pop()
             elif len(potential_men) > 1:
                 # If multiple potential men, pick one. This might be arbitrary
                 # but necessary if PDDL types aren't available in the Task object.
                 # Assumes problem structure allows identification or is simple.
                 self.man_name = next(iter(potential_men))
             # If still None, self.man_name remains None. Heuristic might fail later.
             # This indicates a problem in identifying the man object based on initial state facts.
             # Assuming valid problems allow man identification this way.


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        loose_nuts_in_goal = set()
        man_loc = None
        carrying_usable_spanner = False
        usable_spanners_at_loc = set() # Stores (spanner_name, location)

        # Parse state to find relevant facts
        state_facts = {parse_fact(f) for f in state}
        state_fact_strings = set(state) # Keep strings for quick lookups like '(usable s)'

        # Find man location and if carrying spanner
        for fact in state_facts:
            if fact[0] == 'at' and fact[1] == self.man_name:
                man_loc = fact[2]
            elif fact[0] == 'carrying' and fact[1] == self.man_name:
                 # Check if the carried spanner is usable
                 carried_spanner_name = fact[2]
                 if f'(usable {carried_spanner_name})' in state_fact_strings:
                     carrying_usable_spanner = True

        # Find loose goal nuts and their locations
        loose_nut_locations_in_goal = set() # Stores location names
        for fact in state_facts:
            if fact[0] == 'loose' and fact[1] in self.goal_nuts:
                loose_nuts_in_goal.add(fact[1])
                # Find location of this loose nut
                # Iterate through state_facts again to find the 'at' fact for this nut
                nut_name = fact[1]
                nut_loc = None
                for at_fact in state_facts:
                    if at_fact[0] == 'at' and at_fact[1] == nut_name:
                        nut_loc = at_fact[2]
                        break # Found location for this nut
                if nut_loc:
                    loose_nut_locations_in_goal.add(nut_loc)
                # Note: If a loose goal nut is not 'at' any location in the state,
                # it won't be added to loose_nut_locations_in_goal. This is handled
                # by checking if loose_nut_locations_in_goal is empty later.


        # Find usable spanners at locations
        usable_spanner_names_at_loc = set()
        for fact in state_facts:
             # Check if fact is (usable s) and man is NOT carrying s
             if fact[0] == 'usable':
                 spanner_name = fact[1]
                 if f'(carrying {self.man_name} {spanner_name})' not in state_fact_strings:
                      usable_spanner_names_at_loc.add(spanner_name)

        for spanner_name in usable_spanner_names_at_loc:
             # Find location of this usable spanner
             spanner_loc = None
             for fact in state_facts:
                  if fact[0] == 'at' and fact[1] == spanner_name:
                       spanner_loc = fact[2]
                       break # Found location for this spanner
             if spanner_loc:
                  usable_spanners_at_loc.add((spanner_name, spanner_loc))
             # Note: If a usable spanner is not 'at' any location in the state,
             # it won't be added to usable_spanners_at_loc.


        N_loose_in_goal = len(loose_nuts_in_goal)

        # Heuristic calculation
        h = N_loose_in_goal

        if h == 0:
            return 0 # Goal reached

        # Ensure man_loc is valid (should be if problem is well-formed and man was identified)
        if man_loc is None or man_loc not in self.distances:
             # Man's location is unknown or not in the graph - indicates problem parsing or unsolvability
             return float('inf') # Cannot compute distance

        # Add cost to get to the "action zone"
        if carrying_usable_spanner:
            # Man has a spanner, needs to go to a nut
            if not loose_nut_locations_in_goal:
                 # This implies there are loose goal nuts (h > 0), but none of them
                 # have an '(at nut loc)' fact in the state, or their location
                 # is not in the graph. Likely unsolvable or malformed state.
                 return float('inf')

            min_dist_to_nut = math.inf
            for l_n in loose_nut_locations_in_goal:
                 # Check if distance is computable (location exists in graph)
                 if l_n in self.distances[man_loc]:
                      min_dist_to_nut = min(min_dist_to_nut, self.distances[man_loc][l_n])

            if min_dist_to_nut == math.inf:
                 # Nearest nut location is unreachable from man_loc
                 return float('inf')

            h += min_dist_to_nut

        else: # Not carrying usable spanner
            # Man needs to get a spanner
            if not usable_spanners_at_loc:
                # No usable spanners available at locations
                return float('inf') # Unsolvable from this state

            min_cost_to_spanner_pickup = math.inf
            for s, l_s in usable_spanners_at_loc:
                 # Check if distance is computable (location exists in graph)
                 if l_s in self.distances[man_loc]:
                      cost_to_spanner_pickup = self.distances[man_loc][l_s] + 1 # Travel + pickup
                      min_cost_to_spanner_pickup = min(min_cost_to_spanner_pickup, cost_to_spanner_pickup)

            if min_cost_to_spanner_pickup == math.inf:
                 # Nearest spanner location is unreachable from man_loc
                 return float('inf')

            h += min_cost_to_spanner_pickup

        return h
