from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Compute shortest path distances from start_node to all reachable nodes
    in an unweighted graph using BFS.
    """
    distances = {node: float('inf') for node in graph}
    if start_node in distances: # Ensure start_node is in the graph nodes
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph: # Handle nodes with no outgoing links
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all required nuts by summing the costs
    of sequentially getting a usable spanner and taking it to a loose nut location.
    Uses shortest path distances on the location graph.
    Assumes all goal nuts are at the same location.
    Assumes the man can carry at most one spanner at a time.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing
        shortest paths, and identifying goal nuts and their locations.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Build location graph from link facts
        self.location_graph = defaultdict(list)
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1) # Assuming links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Add locations mentioned in initial state or goals even if not linked
        # This ensures all relevant locations are included in the distance calculation
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.all_locations.add(loc)
        for goal in self.goals:
             # Goals are typically (tightened nut), but could potentially be (at obj loc)
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 self.all_locations.add(loc)

        # Ensure all locations from graph keys/values are in the set
        self.all_locations.update(self.location_graph.keys())
        for neighbors in self.location_graph.values():
             self.all_locations.update(neighbors)

        # Compute all-pairs shortest paths
        self.distances = {}
        for loc in self.all_locations:
            self.distances[loc] = bfs(self.location_graph, loc)

        # Identify goal nuts and their required location
        self.goal_nuts = set()
        self.goal_nut_location = None
        # Assuming all goal nuts are at the same location, find the location of the first one
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.goal_nuts.add(nut)
                # Find the location of this nut from the initial state (assuming nuts are static)
                # This assumes the nut exists and is placed in the initial state
                for fact in initial_state:
                    if match(fact, "at", nut, "*"):
                        _, _, loc = get_parts(fact)
                        self.goal_nut_location = loc
                        break # Found location for one goal nut, assume it's the same for others
            # If goal is (at obj loc), we don't handle that for this spanner heuristic

        # If goal nuts are specified but their location wasn't found in init,
        # the heuristic will return infinity later.

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # loc1 is not a known location or loc2 is unreachable from loc1
             return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify man's location
        man_location = None
        carried_spanner = None
        usable_spanners_on_ground = defaultdict(list) # loc -> list of usable spanners
        loose_nuts = set()

        # Parse state facts
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Assuming 'bob' is the man based on example instances
                if obj == 'bob':
                     man_location = loc
                # We don't need locations of nuts/spanners if they are carried,
                # only spanners on the ground. Nuts are static, their location is from init.

            elif match(fact, "carrying", "*", "*"):
                 _, m, s = get_parts(fact)
                 # Assuming one man ('bob') and he can carry at most one spanner
                 if m == 'bob':
                    carried_spanner = s

            elif match(fact, "usable", "*"):
                 _, s = get_parts(fact)
                 # This spanner 's' is usable. Is it on the ground or carried?
                 is_on_ground = False
                 spanner_loc = None
                 for fact_at in state:
                     if match(fact_at, "at", s, "*"):
                         _, _, spanner_loc = get_parts(fact_at)
                         is_on_ground = True
                         break
                 if is_on_ground:
                     usable_spanners_on_ground[spanner_loc].append(s)
                 # If not on ground, it must be carried. Check if the carried spanner is this one.
                 # The 'carried_spanner' variable already tracks the name if any is carried.
                 # We check if the carried spanner is usable below.

            elif match(fact, "loose", "*"):
                 _, nut = get_parts(fact)
                 loose_nuts.add(nut)

        # Identify required nuts (loose nuts that are goal nuts)
        required_nuts = [nut for nut in loose_nuts if nut in self.goal_nuts]
        num_required_nuts = len(required_nuts)

        # Identify usable spanners available (on ground + carried if usable)
        available_usable_spanners = [] # List of (spanner_name, location) tuples
        for loc, spanners in usable_spanners_on_ground.items():
             available_usable_spanners.extend([(s, loc) for s in spanners])

        carried_spanner_is_usable = False
        if carried_spanner:
             # Check if the carried spanner is also listed as usable in the state
             if f"(usable {carried_spanner})" in state:
                 carried_spanner_is_usable = True
                 # Add carried usable spanner to available list, located at man's location
                 # This allows the greedy selection process to consider the carried spanner
                 # as being "at the man's location" for the first trip.
                 available_usable_spanners.append((carried_spanner, man_location))


        num_available_usable_spanners = len(available_usable_spanners)

        # Check for unsolvability: Not enough usable spanners or goal location unknown/unreachable
        if num_required_nuts > num_available_usable_spanners:
            return float('inf')
        if num_required_nuts > 0 and self.goal_nut_location is None:
             # Goal nuts exist but their location wasn't found in init. Unsolvable.
             return float('inf')
        if num_required_nuts > 0 and man_location is None:
             # Nuts need tightening but man location is unknown. Unsolvable.
             return float('inf')


        # If no nuts need tightening, goal is reached
        if num_required_nuts == 0:
            return 0

        # --- Heuristic Calculation (Greedy Trip Estimation) ---
        h = 0
        current_man_loc = man_location
        remaining_nuts_to_tighten = num_required_nuts
        # Use a list of available spanners that we can remove from
        remaining_available_spanners = list(available_usable_spanners)

        # Assume all required nuts are at the same goal location
        goal_nut_loc = self.goal_nut_location

        # Handle the case where the man starts carrying a usable spanner
        # This spanner is used for the first nut without a pickup action
        if carried_spanner_is_usable:
             # Cost is walk from man's current location to nut location + tighten
             walk_cost = self.get_distance(current_man_loc, goal_nut_loc)
             if walk_cost == float('inf'): return float('inf') # Cannot reach nut location

             h += walk_cost + 1 # 1 for tighten_nut

             current_man_loc = goal_nut_loc
             remaining_nuts_to_tighten -= 1

             # Remove the carried spanner from the list of available spanners
             # It was added with location = man_location
             item_to_remove = None
             for item in remaining_available_spanners:
                 if item[0] == carried_spanner and item[1] == man_location:
                     item_to_remove = item
                     break
             if item_to_remove:
                 remaining_available_spanners.remove(item_to_remove)
             # else: Should not happen if carried_spanner_is_usable is true and it was added


        # Handle remaining nuts requiring picking up a spanner
        while remaining_nuts_to_tighten > 0:
            # Find the spanner location L_S among remaining available spanners
            # that minimizes dist(CurrentManLoc, L_S) + dist(L_S, GoalNutLoc)
            min_walk_for_next_trip = float('inf')
            best_spanner_loc_for_next_trip = None

            # Need to iterate over unique locations that still have available spanners
            locations_with_available_spanners = {loc for s, loc in remaining_available_spanners}

            for spanner_loc in locations_with_available_spanners:
                 dist_to_spanner = self.get_distance(current_man_loc, spanner_loc)
                 dist_spanner_to_nut = self.get_distance(spanner_loc, goal_nut_loc)

                 if dist_to_spanner == float('inf') or dist_spanner_to_nut == float('inf'):
                     continue # Cannot reach this spanner location or cannot take it to the nut

                 walk_cost = dist_to_spanner + dist_spanner_to_nut
                 if walk_cost < min_walk_for_next_trip:
                     min_walk_for_next_trip = walk_cost
                     best_spanner_loc_for_next_trip = spanner_loc

            # If no reachable usable spanner location exists among remaining, problem is unsolvable
            if best_spanner_loc_for_next_trip is None:
                 return float('inf')

            # Cost for this trip: walk + pickup + tighten
            trip_cost = min_walk_for_next_trip + 1 + 1 # 1 for pickup, 1 for tighten
            h += trip_cost

            # Update state for the next iteration
            current_man_loc = goal_nut_loc
            remaining_nuts_to_tighten -= 1

            # Remove one spanner from the chosen location
            spanner_to_remove = None
            for i in range(len(remaining_available_spanners)):
                 if remaining_available_spanners[i][1] == best_spanner_loc_for_next_trip:
                     spanner_to_remove = remaining_available_spanners[i]
                     break
            if spanner_to_remove:
                 remaining_available_spanners.remove(spanner_to_remove)
            # else: Should not happen based on initial solvability check and loop condition

        return h
