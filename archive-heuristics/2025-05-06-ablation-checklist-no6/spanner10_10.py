from collections import deque
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all required nuts by greedily selecting
    the next nut to tighten based on minimum travel and spanner acquisition cost,
    accounting for multiple carried spanners.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal nuts (those that need to be tightened).
        - Location graph and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Extract all locations from initial state and static facts
        all_locations = set()
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3: # (at obj loc)
                     all_locations.add(parts[2])
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # (link loc1 loc2)
                    all_locations.add(parts[1])
                    all_locations.add(parts[2])

        # 2. Build location graph from static link facts
        self.location_graph = {loc: set() for loc in all_locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.location_graph[l1].add(l2)
                self.location_graph[l2].add(l1) # Links are bidirectional

        # 3. Precompute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.location_graph:
            self.distances[start_loc] = self._bfs(start_loc)

        # 4. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all reachable nodes."""
        distances = {node: float('inf') for node in self.location_graph}
        if start_node in distances: # Ensure start_node is in the graph
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                if current_node not in self.location_graph:
                     continue

                for neighbor in self.location_graph.get(current_node, []):
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def dist(self, loc1, loc2):
        """Get the precomputed shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # Locations might be disconnected or not exist in the graph
             return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location and name
        man_name = None
        man_location = None
        all_spanners_in_state = set()
        all_nuts_in_state = set()

        # Identify all spanners and nuts present in the current state facts
        for fact in state:
            parts = get_parts(fact)
            if len(parts) > 1:
                predicate = parts[0]
                obj_name = parts[1]
                if obj_name.startswith("spanner"):
                    all_spanners_in_state.add(obj_name)
                elif obj_name.startswith("nut"):
                    all_nuts_in_state.add(obj_name)

        # Find the object at a location that is not a spanner or nut
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_at, loc_at = get_parts(fact)[1:]
                if obj_at not in all_spanners_in_state and obj_at not in all_nuts_in_state:
                    man_name = obj_at
                    man_location = loc_at
                    break # Found the man

        # Fallback: If man not found by location, check 'carrying'
        if man_name is None:
             for fact in state:
                 if match(fact, "carrying", "*", "*"):
                     man_name = get_parts(fact)[1]
                     # Need to find his location
                     for fact_at in state:
                         if match(fact_at, "at", man_name, "*"):
                             man_location = get_parts(fact_at)[2]
                             break
                     break # Found the man

        if man_name is None or man_location is None:
             # Man object or his location not found in state - invalid state?
             return float('inf')


        # 2. Identify untightened nuts and their locations
        untightened_nuts = set()
        nut_locations = {} # {nut_name: location}
        for goal_nut in self.goal_nuts:
            is_tightened = False
            for fact in state:
                if match(fact, "tightened", goal_nut):
                    is_tightened = True
                    break
            if not is_tightened:
                untightened_nuts.add(goal_nut)
                # Find the location of this nut in the current state
                nut_loc = None
                for fact in state:
                    if match(fact, "at", goal_nut, "*"):
                        nut_loc = get_parts(fact)[2]
                        break
                if nut_loc:
                    nut_locations[goal_nut] = nut_loc
                else:
                    # Nut needing tightening has no location in state - unsolvable
                    return float('inf')


        # If all goal nuts are tightened, heuristic is 0
        if not untightened_nuts:
            return 0

        # 3. Identify usable spanners carried by the man and at locations
        carried_usable_spanners = set() # {spanner_name}
        usable_spanners_at_locs = {} # {spanner_name: location}

        for fact in state:
            if match(fact, "usable", "*"):
                spanner_name = get_parts(fact)[1]
                # Check if the man is carrying this spanner
                is_carried = False
                for carrying_fact in state:
                    if match(carrying_fact, "carrying", man_name, spanner_name):
                        is_carried = True
                        carried_usable_spanners.add(spanner_name)
                        break

                if not is_carried:
                    # Find location of this usable spanner
                    spanner_loc = None
                    for fact_at in state:
                        if match(fact_at, "at", spanner_name, "*"):
                            spanner_loc = get_parts(fact_at)[2]
                            break
                    if spanner_loc:
                        usable_spanners_at_locs[spanner_name] = spanner_loc
                    # If spanner_loc is None, the usable spanner is not at a location
                    # and not carried by the man. Treat as unavailable.


        # Check if enough usable spanners exist in total
        num_needed_spanners = len(untightened_nuts)
        num_available_total = len(carried_usable_spanners) + len(usable_spanners_at_locs)
        if num_available_total < num_needed_spanners:
             # Not enough spanners in the entire state to tighten all nuts
             return float('inf')


        # --- Greedy Heuristic Calculation ---
        total_cost = 0
        curr_loc = man_location
        current_carried_usable_count = len(carried_usable_spanners)
        current_available_spanner_locations = dict(usable_spanners_at_locs) # Make a mutable copy {s_name: s_loc}

        remaining_nuts = set(untightened_nuts)

        while remaining_nuts:
            min_cost_for_next_nut = float('inf')
            best_nut = None
            spanner_name_to_pickup = None # Track which spanner to pick up if needed for the best nut

            # Iterate through remaining nuts to find the best next one
            for nut in remaining_nuts:
                nut_loc = nut_locations[nut]
                cost_to_tighten_this_nut = 0
                temp_spanner_name_to_pickup = None # Local variable for this nut's calculation

                if current_carried_usable_count > 0:
                    # Use a carried spanner
                    cost_to_tighten_this_nut = self.dist(curr_loc, nut_loc) + 1 # travel + tighten
                else:
                    # Need to acquire a spanner
                    closest_spanner_loc = None
                    closest_spanner_name = None
                    min_dist_to_spanner = float('inf')

                    # Find the closest available spanner at a location
                    for s_name, s_loc in current_available_spanner_locations.items():
                         dist_to_s = self.dist(curr_loc, s_loc)
                         if dist_to_s < min_dist_to_spanner:
                             min_dist_to_spanner = dist_to_s
                             closest_spanner_loc = s_loc
                             closest_spanner_name = s_name

                    if closest_spanner_loc is None or min_dist_to_spanner == float('inf'):
                        # No usable spanners available at locations.
                        # Since total availability was checked, this implies the remaining
                        # nuts require more spanners than are currently available at locations
                        # plus carried. This path is not feasible.
                        return float('inf')

                    # Cost includes travel to spanner, pickup, travel to nut, tighten
                    cost_to_tighten_this_nut = min_dist_to_spanner + 1 + self.dist(closest_spanner_loc, nut_loc) + 1 # travel_s + pickup + travel_n + tighten
                    temp_spanner_name_to_pickup = closest_spanner_name # Remember which spanner to pick up

                # Check if this nut is the best option so far
                if cost_to_tighten_this_nut < min_cost_for_next_nut:
                    min_cost_for_next_nut = cost_to_tighten_this_nut
                    best_nut = nut
                    spanner_name_to_pickup = temp_spanner_name_to_pickup # Store the spanner name if one was needed

            # Found the best nut to tighten next
            if best_nut is None:
                 # Should not happen if remaining_nuts is not empty and solvable
                 return float('inf') # Error state or unsolvable

            # Add the cost for this step
            total_cost += min_cost_for_next_nut

            # Update the state simulation for the next iteration
            chosen_nut_loc = nut_locations[best_nut]

            if current_carried_usable_count == 0:
                # Man picked up a spanner (spanner_name_to_pickup) and traveled to the nut
                # Man ends up at the nut location.
                curr_loc = chosen_nut_loc
                # The spanner picked up is used immediately for this nut.
                # It becomes unusable. Carried count remains 0 for the start of the next iteration.
                # Remove the picked up spanner from available ones at locations.
                if spanner_name_to_pickup and spanner_name_to_pickup in current_available_spanner_locations:
                     del current_available_spanner_locations[spanner_name_to_pickup]
            else: # current_carried_usable_count > 0
                # Man used a carried spanner and traveled to the nut
                curr_loc = chosen_nut_loc
                current_carried_usable_count -= 1 # One carried spanner used up

            # Remove the tightened nut from the remaining list
            remaining_nuts.remove(best_nut)

        return total_cost
