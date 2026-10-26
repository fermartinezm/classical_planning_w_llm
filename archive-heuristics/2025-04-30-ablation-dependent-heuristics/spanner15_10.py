import math

class spannerHeuristic:
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal (tightening all goal nuts)
    by summing up the estimated costs for individual components:
    1. The number of 'tighten_nut' actions required (equal to the number of loose goal nuts).
    2. The minimum travel cost for the man to reach the location of the first loose goal nut.
    3. The minimum cost (travel + pickup) for the man to acquire the first usable spanner, if he is not already carrying one.
    4. A fixed estimated cost for each subsequent loose goal nut, representing the actions (pickup + tighten) needed after the first one, simplifying travel costs between subsequent nuts and spanners.

    Assumptions:
    - The problem instance is solvable (unless the heuristic returns infinity).
    - The graph of locations defined by 'link' predicates is connected for all locations relevant to the problem (man's start location, nut locations, spanner locations).
    - There is exactly one man object.
    - Nut locations are static and defined in the initial state.
    - Spanner locations are dynamic (either at a location or carried).
    - Object types (man, spanner, nut, location) can be inferred from initial state facts or task facts.

    Heuristic Initialization:
    The constructor precomputes static information:
    - Parses 'link' facts to build a graph of locations.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies the man object, spanner objects, goal nut objects, and their initial locations (for nuts).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of goal nuts that are currently in a 'loose' state. If this set is empty, the goal is reached, and the heuristic value is 0.
    2. Find the current location of the man. If the man's location is not found in the state, the state is likely invalid or unsolvable, return infinity.
    3. Determine the locations of the loose goal nuts.
    4. Calculate the minimum shortest path distance from the man's current location to any of the loose goal nut locations. If no nut location is reachable, return infinity. This distance estimates the travel cost to the first nut.
    5. Check if the man is currently carrying a usable spanner.
    6. Identify usable spanners that are currently located at specific locations (not carried). Determine their locations.
    7. If the man is not carrying a usable spanner:
       a. Calculate the minimum shortest path distance from the man's current location to any location containing a usable spanner.
       b. If no usable spanners are found at any reachable location, and the man is not carrying one, the problem is unsolvable from this state. Return infinity.
       c. The estimated cost to acquire the first spanner is this minimum distance plus 1 (for the 'pickup_spanner' action).
    8. Initialize the heuristic value with the number of loose goal nuts (representing the 'tighten_nut' actions).
    9. Add the minimum distance calculated in step 4 (travel to the first nut).
    10. If the man was not carrying a usable spanner, add the spanner acquisition cost calculated in step 7c.
    11. For each loose goal nut beyond the first one (i.e., number of loose nuts - 1), add a fixed cost (e.g., 2). This fixed cost approximates the actions needed for subsequent nuts (pickup + tighten), simplifying the estimation of additional travel between nuts and spanners. A value of 2 is chosen as a simple lower bound for these two actions.
    12. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.infinity = float('inf')

        # 1. Parse object types from task facts
        self.obj_types = {}
        for fact in task.facts:
            if fact.startswith('(') and fact.endswith(')'):
                parts = fact.strip('()').split()
                # Heuristic: Assume facts with 2 parts where the first is lowercase are type facts
                if len(parts) == 2 and parts[0].islower():
                     self.obj_types[parts[1]] = parts[0]

        # 2. Identify the man object (assume one man)
        self.man_name = None
        for obj, obj_type in self.obj_types.items():
            if obj_type == 'man':
                self.man_name = obj
                break
        if self.man_name is None:
             # Fallback: Try to find man in initial state 'at' facts
             for fact in task.initial_state:
                  if fact.startswith('(at '):
                       parts = fact.strip('()').split()
                       if len(parts) == 3 and parts[1] in self.obj_types and self.obj_types[parts[1]] == 'man':
                            self.man_name = parts[1]
                            break
             if self.man_name is None:
                print("Warning: Could not identify the man object.") # Should not happen in valid problems

        # 3. Identify spanner objects
        self.spanners = {obj for obj, obj_type in self.obj_types.items() if obj_type == 'spanner'}

        # 4. Identify goal nuts and their initial locations
        self.goal_nuts = set()
        for goal_fact in task.goals:
            if goal_fact.startswith('(tightened '):
                nut_name = goal_fact.strip('()').split()[1]
                self.goal_nuts.add(nut_name)

        self.nut_locations = {} # nut -> location (static)
        for fact in task.initial_state:
             if fact.startswith('(at '):
                  parts = fact.strip('()').split()
                  obj_name = parts[1]
                  loc_name = parts[2]
                  if obj_name in self.goal_nuts:
                       self.nut_locations[obj_name] = loc_name

        # 5. Parse static facts for location graph
        self.location_graph = {}
        self.locations = set()
        for fact in task.static:
            if fact.startswith('(link '):
                parts = fact.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Add any locations mentioned in initial state/goal if not in links (e.g., shed, gate)
        for fact in task.initial_state:
             if fact.startswith('(at '):
                  parts = fact.strip('()').split()
                  if len(parts) == 3:
                       self.locations.add(parts[2]) # Add location
        for nut in self.goal_nuts:
             if nut in self.nut_locations:
                  self.locations.add(self.nut_locations[nut])

        # Ensure all locations in the graph structure exist as keys, even if they have no links initially
        for loc in self.locations:
             self.location_graph.setdefault(loc, set())


        # 6. Compute all-pairs shortest paths using BFS
        self.dist = {}
        all_locs_list = list(self.locations) # Ensure consistent order
        for start_loc in all_locs_list:
            self.dist[start_loc] = {}
            q = [(start_loc, 0)]
            visited = {start_loc}
            while q:
                (curr_loc, d) = q.pop(0)
                self.dist[start_loc][curr_loc] = d
                if curr_loc in self.location_graph:
                    for neighbor in self.location_graph[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # Fill in unreachable distances with infinity
        for l1 in all_locs_list:
            for l2 in all_locs_list:
                if l2 not in self.dist.get(l1, {}):
                     self.dist.setdefault(l1, {})[l2] = self.infinity


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal.
        """
        # 1. Identify loose goal nuts in the current state
        loose_goal_nuts = {n for n in self.goal_nuts if '(loose ' + n + ')' in state}

        if not loose_goal_nuts:
            return 0 # Goal reached

        # 2. Find the man's current location
        man_loc = None
        if self.man_name:
            for fact in state:
                if fact.startswith('(at ' + self.man_name + ' '):
                    parts = fact.strip('()').split()
                    if len(parts) == 3:
                         man_loc = parts[2]
                         break

        if man_loc is None or man_loc not in self.locations:
             # Man's location is unknown or not in the known graph - unsolvable
             return self.infinity

        # 3. Find locations of loose goal nuts
        nut_locs = {self.nut_locations[n] for n in loose_goal_nuts if n in self.nut_locations}

        # If any loose goal nut location is not in the known graph, unsolvable
        if any(loc not in self.locations for loc in nut_locs):
             return self.infinity

        # 4. Compute minimum distance from man to any loose nut location
        min_dist_to_nut = self.infinity
        for loc_n in nut_locs:
            if man_loc in self.dist and loc_n in self.dist[man_loc]:
                 min_dist_to_nut = min(min_dist_to_nut, self.dist[man_loc][loc_n])

        # If no loose nut location is reachable, unsolvable
        if min_dist_to_nut == self.infinity:
             return self.infinity

        # 5. Check if man is carrying a usable spanner
        man_carrying_usable = False
        if self.man_name:
            for s in self.spanners:
                 if '(carrying ' + self.man_name + ' ' + s + ')' in state and '(usable ' + s + ')' in state:
                      man_carrying_usable = True
                      break

        # 6. Find usable spanners at locations and their locations
        usable_spanners_at_locs = {s for s in self.spanners if '(usable ' + s + ')' in state and any(f.startswith('(at ' + s + ' ') for f in state)}
        spanner_locs = {fact.strip('()').split()[2] for s in usable_spanners_at_locs for fact in state if fact.startswith('(at ' + s + ' ')}

        # If any spanner location is not in the known graph, treat as unreachable for now
        spanner_locs = {loc for loc in spanner_locs if loc in self.locations}


        # 7. Compute cost to get the first spanner if needed
        spanner_cost = 0
        if not man_carrying_usable:
            min_dist_to_spanner = self.infinity
            if spanner_locs:
                for loc_s in spanner_locs:
                     if man_loc in self.dist and loc_s in self.dist[man_loc]:
                          min_dist_to_spanner = min(min_dist_to_spanner, self.dist[man_loc][loc_s])

            # If man needs a spanner but none are reachable at locations, unsolvable
            if min_dist_to_spanner == self.infinity:
                 # Check if any usable spanners exist at all (carried by man was false, at locations unreachable)
                 total_usable_spanners_count = len({s for s in self.spanners if '(usable ' + s + ')' in state})
                 if total_usable_spanners_count == 0:
                      return self.infinity # No usable spanners anywhere
                 else:
                      # Usable spanners exist but are unreachable from man's current location
                      return self.infinity # Unsolvable from here

            spanner_cost = min_dist_to_spanner + 1 # Travel to spanner + pickup action

        # 8. Calculate heuristic
        # Base cost: tighten actions for all loose nuts
        h = len(loose_goal_nuts)

        # Add cost for the first nut: travel to nut + spanner acquisition (if needed)
        h += min_dist_to_nut
        h += spanner_cost

        # Add estimated cost for subsequent nuts (tighten + pickup), ignoring travel between them
        if len(loose_goal_nuts) > 1:
             h += (len(loose_goal_nuts) - 1) * 2 # +1 for tighten, +1 for pickup

        return h

# Example usage (assuming Task object is available):
# task = ... # Load your task here
# heuristic = spannerHeuristic(task)
# state = ... # Get a state here
# h_value = heuristic(state)
# print(f"Heuristic value: {h_value}")
