from heuristics.heuristic_base import Heuristic
from collections import deque
import sys # Import sys for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions (walk, pickup_spanner, tighten_nut)
    required to tighten all loose nuts. It uses a greedy simulation approach,
    prioritizing acquiring a usable spanner if needed, and then moving to and
    tightening the nearest loose nut. The simulation assumes the man will always
    choose the nearest available resource (usable ground spanner or loose nut)
    when needed.

    # Assumptions
    - Nuts are static and their locations do not change throughout the plan.
    - Spanners become unusable after one use (`tighten_nut` action).
    - Spanners cannot be dropped or made usable again.
    - The man can carry multiple spanners simultaneously.
    - There is exactly one man object in the domain.
    - Enough usable spanners exist initially (either carried or on the ground)
      to tighten all loose nuts in a solvable problem instance. If not, the
      heuristic returns infinity.
    - All locations mentioned in facts are part of the graph defined by `link` facts,
      or are isolated locations reachable only by being there initially.

    # Heuristic Initialization
    - Precomputes shortest path distances between all pairs of locations based on
      `link` facts using Breadth-First Search (BFS). This graph is built from
      `link` facts in the static information. Locations mentioned in initial/goal
      states but not in links are treated as isolated nodes for distance calculation.
    - Identifies the man object, all nut objects, and all spanner objects by
      parsing the initial state and goal state facts.
    - Stores the static locations of nuts, assuming they do not move.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state information: the man's current location, the set
       of usable spanners currently carried by the man, a dictionary of usable
       spanners on the ground mapping spanner object to its location, and a list
       of all loose nuts.
    2. If the list of loose nuts is empty, the goal state has been reached, so
       return 0.
    3. Initialize the total estimated cost to 0.
    4. Initialize the state variables for the greedy simulation: the man's current
       location (starting from the actual state location), the current count of
       usable spanners carried, a mutable list of remaining loose nuts, and a
       mutable dictionary of remaining usable ground spanners and their locations.
    5. Enter a loop that continues as long as there are remaining loose nuts to
       tighten:
       a. Check if the man currently carries any usable spanners (i.e., if
          `current_carried_usable` is 0). A usable spanner is required for the
          next `tighten_nut` action.
       b. If the man has no usable spanners carried (`current_carried_usable == 0`):
          i. Search through the `remaining_ground_spanners_info` to find the
             usable spanner on the ground that is nearest to the man's current
             location. Calculate the shortest distance using the precomputed map.
          ii. If no usable ground spanners are left (or all are unreachable),
              the problem is unsolvable from this state, return `float('inf')`.
          iii. Calculate the cost to reach this nearest spanner and pick it up.
              This cost is the shortest distance to the spanner's location plus 1
              action for `pickup_spanner`. Add this cost to the `total_cost`.
          iv. Update the man's `current_man_location` to the location of the
              picked-up spanner.
          v. Increment `current_carried_usable` by 1.
          vi. Remove the picked-up spanner from the `remaining_ground_spanners_info`.
       c. Now that the man is simulated to have at least one usable spanner:
          i. Search through the `remaining_loose_nuts` to find the loose nut
             that is nearest to the man's current location. Calculate the shortest
             distance using the precomputed map and the nut's static location.
          ii. If no reachable loose nuts are left, but the list is not empty,
              return `float('inf')` (implies remaining nuts are unreachable).
          iii. Calculate the cost to reach this nearest nut and tighten it.
               This cost is the shortest distance to the nut's location plus 1
               action for `tighten_nut`. Add this cost to the `total_cost`.
          iv. Update the man's `current_man_location` to the location of the
              tightened nut.
          v. Decrement `current_carried_usable` by 1 (as one spanner is used up).
          vi. Remove the tightened nut from the `remaining_loose_nuts` list.
    6. Once the `remaining_loose_nuts` list is empty, the simulation is complete.
       Return the final `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing
        shortest paths.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to find objects and nut locations

        # Build location graph from link facts
        self.location_graph = {}
        all_locations = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional
                all_locations.add(l1)
                all_locations.add(l2)

        # Add locations from initial/goal state that might not be in links (isolated)
        for fact in self.initial_state | self.goals:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 loc = parts[2]
                 all_locations.add(loc)

        # Precompute shortest paths using BFS
        self.shortest_paths = {}
        for start_loc in all_locations:
             # Only run BFS if the location is part of the graph (has links)
             # Otherwise, it's an isolated location, distance is only 0 to itself.
             if start_loc in self.location_graph:
                 self.shortest_paths[start_loc] = self._bfs(start_loc)
             else:
                 # Isolated location, only reachable from itself
                 self.shortest_paths[start_loc] = {start_loc: 0}


        # Identify objects: man, nuts, spanners
        self.man_obj = None
        self.nut_objects = set()
        self.spanner_objects = set()

        # Identify nuts from initial/goal state (loose or tightened)
        for fact in self.initial_state | self.goals:
            parts = get_parts(fact)
            if parts and parts[0] in ["loose", "tightened"] and len(parts) == 2:
                self.nut_objects.add(parts[1])

        # Identify spanners and man from initial state 'at' and 'carrying' facts
        locatable_objects = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                 obj = parts[1]
                 locatable_objects.add(obj)
             elif parts and parts[0] == "carrying" and len(parts) == 3:
                  man_obj = parts[1]
                  spanner_obj = parts[2]
                  locatable_objects.add(man_obj)
                  locatable_objects.add(spanner_obj)

        # Try to identify spanners more specifically (e.g., from usable/carrying predicates)
        identified_spanners = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "usable" and len(parts) == 2:
                 identified_spanners.add(parts[1])
             elif parts and parts[0] == "carrying" and len(parts) == 3:
                  identified_spanners.add(parts[2]) # The carried object is a spanner

        self.spanner_objects = identified_spanners

        # The man is the remaining locatable object that isn't a nut or identified spanner
        potential_men = locatable_objects - self.nut_objects - self.spanner_objects
        if len(potential_men) == 1:
             self.man_obj = list(potential_men)[0]
        elif len(potential_men) > 1:
             # Fallback: If multiple candidates, try finding the one in a 'carrying' predicate
             man_candidate = None
             for fact in self.initial_state:
                  parts = get_parts(fact)
                  if parts and parts[0] == "carrying" and len(parts) == 3:
                       man_candidate = parts[1]
                       break
             if man_candidate and man_candidate in potential_men:
                  self.man_obj = man_candidate
             else:
                  # If still ambiguous, pick one. Assuming the first potential man is the man.
                  if potential_men:
                       self.man_obj = list(potential_men)[0]
                  else:
                       # This case is problematic, implies no man found.
                       # Relying on 'bob' assumption from example state as a last resort.
                       # This should ideally not happen in valid problem instances.
                       self.man_obj = 'bob' # Default assumption if parsing fails


        # Store nut locations (assuming static)
        self.nut_locations = {}
        for nut in self.nut_objects:
             # Find the 'at' fact for this nut in the initial state
             found_loc = False
             for fact in self.initial_state:
                  parts = get_parts(fact)
                  if parts and parts[0] == "at" and len(parts) == 3 and parts[1] == nut:
                       self.nut_locations[nut] = parts[2]
                       found_loc = True
                       break
             if not found_loc:
                  # This nut doesn't have an initial location? Problematic instance.
                  # Mark as location unknown, will be treated as unreachable later.
                  self.nut_locations[nut] = None


    def _bfs(self, start_node):
        """Performs BFS to find shortest distances from start_node."""
        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft() # Use popleft for BFS
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Gets the shortest distance between two locations."""
        if loc1 is None or loc2 is None:
             return float('inf') # Cannot calculate distance if location is unknown
        if loc1 == loc2:
            return 0
        # Check if start_loc is in the precomputed paths and if target is reachable
        if loc1 in self.shortest_paths and loc2 in self.shortest_paths.get(loc1, {}):
            return self.shortest_paths[loc1][loc2]
        # If loc1 is an isolated node (not in graph keys but in all_locations),
        # its shortest_paths entry only contains distance to itself.
        # If loc2 is not in the distances from loc1, they are unreachable.
        return float('inf') # Indicate unreachable


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Find man's current location
        man_location = None
        # Ensure man_obj was identified in __init__
        if self.man_obj:
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "at" and len(parts) == 3 and parts[1] == self.man_obj:
                    man_location = parts[2]
                    break

        if man_location is None:
             # Man's location is unknown in this state, problem state is invalid or unreachable
             # In a valid planning problem, the man should always have a location.
             # Returning infinity suggests this state is bad or unreachable.
             return float('inf')

        # Find usable spanners the man is carrying
        carried_usable_spanners = set()
        if self.man_obj: # Only check if man_obj was identified
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "carrying" and len(parts) == 3 and parts[1] == self.man_obj:
                    spanner = parts[2]
                    if f"(usable {spanner})" in state:
                        carried_usable_spanners.add(spanner)

        # Find usable spanners on the ground and their locations
        usable_ground_spanners_info = {} # {spanner_obj: location}
        for spanner in self.spanner_objects:
             if f"(usable {spanner})" in state:
                  # Find the 'at' fact for this spanner in the current state
                  for fact in state:
                       parts = get_parts(fact)
                       if parts and parts[0] == "at" and len(parts) == 3 and parts[1] == spanner:
                            usable_ground_spanners_info[spanner] = parts[2]
                            break


        # Find loose nuts
        loose_nuts = [nut for nut in self.nut_objects if f"(loose {nut})" in state]

        # If no loose nuts, goal is reached, heuristic is 0.
        if not loose_nuts:
            return 0

        # --- Greedy Simulation ---
        current_man_location = man_location
        current_carried_usable = len(carried_usable_spanners)
        remaining_loose_nuts = list(loose_nuts) # Use a list to remove elements
        remaining_ground_spanners_info = dict(usable_ground_spanners_info) # Use a dict copy

        while remaining_loose_nuts:
            # Do we need a spanner for the next tightening action?
            # The simulation proceeds one tighten action at a time. Each requires one usable spanner.
            # If we don't have one carried right now, we must get one from the ground.
            if current_carried_usable == 0:
                # Need to pick up a spanner
                if not remaining_ground_spanners_info:
                    # No usable spanners left anywhere on the ground
                    # This state is likely unsolvable if more nuts need tightening.
                    return float('inf') # Unsolvable from this state

                # Find nearest usable ground spanner
                nearest_spanner = None
                min_dist = float('inf')
                spanner_loc = None
                for s, loc in remaining_ground_spanners_info.items():
                    dist = self.get_distance(current_man_location, loc)
                    if dist == float('inf'):
                         continue # Skip unreachable spanners
                    if dist < min_dist:
                        min_dist = dist
                        nearest_spanner = s
                        spanner_loc = loc

                if nearest_spanner is None:
                     # All remaining ground spanners are unreachable
                     return float('inf') # Unsolvable

                # Add cost to walk to spanner and pick it up
                total_cost += min_dist + 1 # walk + pickup
                current_man_location = spanner_loc
                current_carried_usable += 1
                del remaining_ground_spanners_info[nearest_spanner]

            # Now we have at least one usable spanner, tighten a nut
            # Find nearest loose nut
            nearest_nut = None
            min_dist = float('inf')
            nut_loc = None
            for nut in remaining_loose_nuts:
                loc = self.nut_locations.get(nut) # Use .get for safety
                if loc is None:
                     # Nut location unknown, problematic instance
                     continue # Skip this nut or return inf? Let's skip for now.

                dist = self.get_distance(current_man_location, loc)
                if dist == float('inf'):
                     continue # Skip unreachable nuts

                if dist < min_dist:
                    min_dist = dist
                    nearest_nut = nut
                    nut_loc = loc

            if nearest_nut is None:
                 # No reachable loose nuts left, but remaining_loose_nuts is not empty?
                 # This implies remaining nuts are unreachable.
                 return float('inf') # Unsolvable

            # Add cost to walk to nut and tighten it
            total_cost += min_dist + 1 # walk + tighten
            current_man_location = nut_loc
            current_carried_usable -= 1 # Spanner is used
            remaining_loose_nuts.remove(nearest_nut)

        return total_cost
