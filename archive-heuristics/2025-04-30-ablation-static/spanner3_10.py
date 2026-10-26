import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a given pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for Breadth-First Search
def bfs(start_node, graph):
    """Compute shortest path distances from start_node to all reachable nodes in the graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node might not be in graph if it's an object location not linked
         # but present in initial state. Treat as isolated if no links.
         if start_node in distances:
              distances[start_node] = 0
         return distances # Return distances for just the start node if isolated

    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the travel cost for the man to reach nut locations and spanner locations,
    as well as the cost of pickup and tighten actions. It uses a greedy strategy,
    prioritizing the closest loose goal nut and the closest available usable spanner
    when needed.

    # Assumptions
    - There is only one man.
    - The man can carry only one spanner at a time.
    - A spanner is consumed (becomes unusable) after tightening one nut.
    - The locations of nuts and spanners on the ground are static throughout the plan.
    - The graph of locations connected by links is undirected.

    # Heuristic Initialization
    - Identify the name of the man, all spanners, nuts, and locations by parsing
      the initial state and static facts.
    - Store the initial locations of all nuts and spanners.
    - Identify the set of goal nuts from the task goals.
    - Build the location graph based on `link` predicates in static facts.
    - Compute all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location and whether he is carrying a usable spanner.
    2. Identify all nuts that are currently loose and are goal nuts.
    3. Identify all usable spanners currently on the ground and their locations.
    4. Check if the total number of available usable spanners (carried or on ground)
       is less than the number of loose goal nuts. If so, the problem is likely
       unsolvable from this state (or requires more spanners than exist), return infinity.
    5. Initialize the heuristic cost to 0. Keep track of the man's conceptual
       current location and whether he is conceptually carrying a usable spanner.
       Also, keep track of which usable spanners on the ground have been conceptually
       used for pickup actions.
    6. While there are loose goal nuts remaining:
       a. Select the closest remaining loose goal nut based on the current conceptual
          man location and the precomputed shortest path distances.
       b. Add the travel cost (shortest path distance) from the current conceptual
          man location to the selected nut's location to the heuristic cost.
          Update the man's conceptual location to the nut's location.
       c. If the man is not conceptually carrying a usable spanner:
          i. Find the closest available usable spanner on the ground based on the
             current conceptual man location and the precomputed distances.
          ii. Add the travel cost (shortest path distance) from the current conceptual
              man location to the spanner's location to the heuristic cost.
              Update the man's conceptual location to the spanner's location.
          iii. Add the cost of the `pickup_spanner` action (1) to the heuristic cost.
               Mark this spanner as conceptually used. Update the man's conceptual
               spanner status to 'carrying usable spanner'.
          iv. If the spanner's location was different from the nut's location,
              add the travel cost (shortest path distance) from the spanner's location
              back to the nut's location to the heuristic cost. Update the man's
              conceptual location back to the nut's location.
       d. Add the cost of the `tighten_nut` action (1) to the heuristic cost.
          Update the man's conceptual spanner status to 'not carrying usable spanner'
          (as the spanner is consumed). Remove the selected nut from the set of
          remaining loose goal nuts.
    7. Return the total accumulated heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        initial_state_facts = task.initial_state
        static_facts = task.static

        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Infer object types and initial locations
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.location_names = set()
        self.nut_initial_locations = {}
        self.spanner_initial_locations = {}

        all_facts = set(initial_state_facts) | set(static_facts)

        # First pass: Identify types based on specific predicates
        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and len(parts) == 3:
                self.man_name = parts[1] # Assume one man
                self.spanner_names.add(parts[2])
            elif parts[0] == 'usable' and len(parts) == 2:
                self.spanner_names.add(parts[1])
            elif parts[0] == 'tightened' and len(parts) == 2:
                 self.nut_names.add(parts[1])
            elif parts[0] == 'loose' and len(parts) == 2:
                 self.nut_names.add(parts[1])
            elif parts[0] == 'link' and len(parts) == 3:
                self.location_names.add(parts[1])
                self.location_names.add(parts[2])

        # Second pass: Identify locations from 'at' predicates and ensure all mentioned objects are typed
        # Also collect initial locations for static objects (nuts, spanners)
        for fact in initial_state_facts:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 self.location_names.add(loc)
                 if obj == self.man_name:
                     pass # Man's initial location is dynamic
                 elif obj in self.nut_names:
                     self.nut_initial_locations[obj] = loc
                 elif obj in self.spanner_names:
                     self.spanner_initial_locations[obj] = loc
                 else:
                     # Object type not inferred yet, but it's locatable and at a location
                     # For this domain, it must be man, nut, or spanner.
                     # If it's not one of the inferred types, something is wrong or it's a different locatable type we don't care about.
                     pass

        # Ensure all locations mentioned in links are in location_names
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'link' and len(parts) == 3:
                 self.location_names.add(parts[1])
                 self.location_names.add(parts[2])

        # Add any initial locations of nuts/spanners that weren't linked to the location set
        self.location_names.update(self.nut_initial_locations.values())
        self.location_names.update(self.spanner_initial_locations.values())


        # Build location graph
        self.location_graph = {loc: set() for loc in self.location_names}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 in self.location_graph and l2 in self.location_graph:
                    self.location_graph[l1].add(l2)
                    self.location_graph[l2].add(l1)

        # Compute all-pairs shortest paths
        self.distances = {}
        for loc in self.location_names:
            self.distances[loc] = bfs(loc, self.location_graph)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location and spanner status
        current_man_location = None
        man_carrying_spanner_name = None
        usable_spanners_in_state = set()
        spanner_locations_in_state = {} # Current location of spanners on the ground

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj == self.man_name:
                    current_man_location = loc
                elif obj in self.spanner_names:
                    spanner_locations_in_state[obj] = loc
            elif parts[0] == 'carrying' and len(parts) == 3:
                m, s = parts[1], parts[2]
                if m == self.man_name:
                    man_carrying_spanner_name = s
            elif parts[0] == 'usable' and len(parts) == 2:
                usable_spanners_in_state.add(parts[1])

        man_is_carrying_usable_spanner = (man_carrying_spanner_name is not None) and (man_carrying_spanner_name in usable_spanners_in_state)

        # 2. Identify loose goal nuts in current state
        loose_goal_nuts = {nut for nut in self.goal_nuts if f'(loose {nut})' in state}

        # 3. Identify usable spanners on the ground in current state
        available_usable_spanners_on_ground = {(s, loc) for s, loc in spanner_locations_in_state.items() if s in usable_spanners_in_state}

        # 4. Check if enough spanners exist
        num_nuts_to_tighten = len(loose_goal_nuts)
        num_spanners_on_ground = len(available_usable_spanners_on_ground)
        num_spanners_available = num_spanners_on_ground + (1 if man_is_carrying_usable_spanner else 0)

        if num_spanners_available < num_nuts_to_tighten:
            return float('inf') # Unsolvable from this state

        if num_nuts_to_tighten == 0:
            return 0 # Goal state reached

        # Ensure man's current location is in the distance map
        if current_man_location not in self.distances:
             # This location might be isolated or not linked to anything
             # If the man is at an isolated location, he can't move unless a link exists.
             # If the goal nuts/spanners are also at isolated locations, the problem might be solvable locally.
             # A simple approach: if man_loc is not in distances, return inf unless all remaining tasks are at man_loc.
             all_tasks_at_man_loc = True
             for nut in loose_goal_nuts:
                  if self.nut_initial_locations.get(nut) != current_man_location:
                       all_tasks_at_man_loc = False
                       break
             if not all_tasks_at_man_loc:
                  # If man is isolated and tasks are elsewhere, cannot reach them.
                  return float('inf')
             # If all tasks are at man_loc, we only need pickups/tightens if spanners are also local.
             # This case is complex, let's assume valid states have man_loc in the graph.
             # If it happens, the BFS will return inf for other locations, which is correct.
             pass # Let the BFS result handle unreachable locations


        # 5. Initialize heuristic state
        heuristic_cost = 0
        conceptual_man_location = current_man_location
        conceptual_man_is_carrying_usable_spanner = man_is_carrying_usable_spanner
        remaining_loose_goal_nuts = set(loose_goal_nuts)
        used_spanners_on_ground_tuples = set() # Track spanners used conceptually

        # 6. Greedy loop
        while remaining_loose_goal_nuts:
            # a. Select the closest remaining loose goal nut
            next_nut = None
            min_dist_to_nut = float('inf')
            for nut in remaining_loose_goal_nuts:
                nut_loc = self.nut_initial_locations.get(nut) # Get static nut location
                if nut_loc is None or conceptual_man_location not in self.distances or nut_loc not in self.distances[conceptual_man_location]:
                     # This indicates a problem with location data or graph
                     return float('inf') # Should not happen in valid problems
                dist = self.distances[conceptual_man_location][nut_loc]
                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    next_nut = nut

            if next_nut is None:
                 # Should not happen if remaining_loose_goal_nuts is not empty
                 return float('inf') # Indicate problem

            L_nut = self.nut_initial_locations[next_nut]

            # Cost to get to the nut location
            # If man is already at the nut, distance is 0.
            heuristic_cost += min_dist_to_nut
            conceptual_man_location = L_nut

            # Cost to get a spanner if needed for this nut
            if not conceptual_man_is_carrying_usable_spanner:
                # Find the closest available usable spanner on the ground from the *current* conceptual man location (which is now L_nut)
                closest_spanner_info = None # (spanner, location)
                min_dist_to_spanner = float('inf')

                for (s, l_s) in available_usable_spanners_on_ground:
                    if (s, l_s) not in used_spanners_on_ground_tuples:
                         if conceptual_man_location not in self.distances or l_s not in self.distances[conceptual_man_location]:
                              # Should not happen
                              return float('inf') # Indicate problem
                         dist = self.distances[conceptual_man_location][l_s]
                         if dist < min_dist_to_spanner:
                            min_dist_to_spanner = dist
                            closest_spanner_info = (s, l_s)

                if closest_spanner_info is None:
                     # This should not happen if initial spanner count check was correct
                     return float('inf') # Indicate problem

                s, L_s = closest_spanner_info

                # Cost = Travel to spanner + Pickup + Travel to nut (if needed)
                heuristic_cost += min_dist_to_spanner # Travel from nut loc to spanner loc
                conceptual_man_location = L_s # Update man's conceptual location
                heuristic_cost += 1 # Pickup action
                used_spanners_on_ground_tuples.add((s, L_s))
                conceptual_man_is_carrying_usable_spanner = True # Now carrying a spanner

                # After picking up, need to travel back to the nut location if it wasn't the same
                if conceptual_man_location != L_nut:
                     if conceptual_man_location not in self.distances or L_nut not in self.distances[conceptual_man_location]:
                          # Should not happen
                          return float('inf') # Indicate problem
                     heuristic_cost += self.distances[conceptual_man_location][L_nut]
                     conceptual_man_location = L_nut # Update man's conceptual location


            # Cost to tighten the nut
            heuristic_cost += 1
            conceptual_man_is_carrying_usable_spanner = False # Spanner is consumed
            remaining_loose_goal_nuts.remove(next_nut) # This nut is now conceptually tightened

        return heuristic_cost
