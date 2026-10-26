import fnmatch

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not provided, you might need a simple definition like this:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#
#     def __call__(self, node):
#         raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or return empty list
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions needed to serve all passengers.
    The heuristic sums:
    1. The number of board/depart actions needed for unserved passengers.
    2. The estimated lift movement cost to visit necessary floors (origins for unboarded, destinations for boarded).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger info, and goals.
        """
        super().__init__(task)

        # Build floor mapping from static facts
        self.floor_map = self._build_floor_map(task.static)

        # Store goal served predicates for quick lookup
        # Goals are typically (and (served p1) (served p2) ...)
        self.goal_passengers = {get_parts(goal)[1] for goal in task.goals if match(goal, "served", "*")}

        # Store passenger origins and destinations from initial state
        self.passenger_info = {}
        # Collect all passengers mentioned in initial state (origin, destin, boarded, served)
        # and add goal passengers to ensure we track all relevant ones.
        all_passengers_set = set()

        for fact in task.initial_state:
             if match(fact, "origin", "*", "*"):
                 p, f = get_parts(fact)[1:]
                 if p not in self.passenger_info:
                     self.passenger_info[p] = {}
                 self.passenger_info[p]['origin'] = f
                 all_passengers_set.add(p)
             elif match(fact, "destin", "*", "*"):
                 p, f = get_parts(fact)[1:]
                 if p not in self.passenger_info:
                     self.passenger_info[p] = {}
                 self.passenger_info[p]['destin'] = f
                 all_passengers_set.add(p)
             elif match(fact, "boarded", "*"):
                 p = get_parts(fact)[1]
                 all_passengers_set.add(p)
             elif match(fact, "served", "*"):
                 p = get_parts(fact)[1]
                 all_passengers_set.add(p)

        # Ensure all passengers from goals are included, even if not in initial state facts
        all_passengers_set.update(self.goal_passengers)
        self.all_passengers = list(all_passengers_set) # Store as list

    def _build_floor_map(self, static_facts):
        """
        Builds a mapping from floor name to an integer index based on 'above' facts.
        Assumes floors form a linear chain where (above f_i f_j) means f_i is higher than f_j.
        Identifies immediate 'above' relationships to build the ordered map using BFS.
        """
        above_pairs = set()
        all_floors = set()

        # First pass to get all floors and all above pairs
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                above_pairs.add((f_above, f_below))
                all_floors.add(f_above)
                all_floors.add(f_below)

        if not all_floors:
             return {} # Handle case with no floors or no above facts

        # Find the lowest floor: the floor in all_floors that is not below any other floor
        # This means f is never the second element in any above_pair.
        floors_that_are_below_others = {f_below for f_above, f_below in above_pairs}

        lowest_floor = None
        potential_lowest = all_floors - floors_that_are_below_others

        if len(potential_lowest) == 1:
            lowest_floor = list(potential_lowest)[0]
        elif len(potential_lowest) > 1:
             # Multiple lowest floors? Disconnected components? Assume sorted order.
             lowest_floor = sorted(list(potential_lowest))[0]
             # print(f"Warning: Multiple potential lowest floors found: {potential_lowest}. Choosing {lowest_floor}.")
        elif not potential_lowest and all_floors:
             # This case should only happen if there's only one floor.
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # print("Error: Could not determine unique lowest floor from 'above' facts.")
                 return {}

        if lowest_floor is None:
             return {}

        # Build the immediate_below_graph
        # f_below -> f_above if f_above is immediately above f_below
        # f_above is immediately above f_below if (above f_above f_below) is true
        # AND there is no f_k such that (above f_above f_k) and (above f_k f_below) are true
        immediate_below_graph = {} # f_below -> list of f_above (immediate)

        for f_i, f_j in above_pairs:
            is_immediate = True
            for f_k in all_floors:
                if f_k != f_i and f_k != f_j:
                    if (f_i, f_k) in above_pairs and (f_k, f_j) in above_pairs:
                        is_immediate = False
                        break
            if is_immediate:
                # f_i is immediately above f_j
                # Add edge f_j -> f_i in immediate_below_graph
                if f_j not in immediate_below_graph:
                    immediate_below_graph[f_j] = []
                immediate_below_graph[f_j].append(f_i)

        # Build the floor map using BFS starting from the lowest floor
        floor_map = {}
        queue = [(lowest_floor, 0)]
        visited = {lowest_floor}

        while queue:
            current_floor, current_index = queue.pop(0) # Use pop(0) for BFS
            floor_map[current_floor] = current_index

            # Find floors immediately above current_floor
            # These are floors f_above such that current_floor -> f_above in immediate_below_graph
            next_floors = immediate_below_graph.get(current_floor, [])

            for next_floor in next_floors:
                if next_floor not in visited:
                    visited.add(next_floor)
                    queue.append((next_floor, current_index + 1))

        # Check if all floors were mapped (implies a connected chain)
        # If not, the map is partial, heuristic might be inaccurate for unmapped floors.
        # We proceed with the partial map.
        # if len(floor_map) != len(all_floors):
        #      print("Warning: Not all floors were mapped. 'above' facts may not form a single chain.")

        return floor_map


    def __call__(self, node):
        """Estimate the required number of actions to reach a goal state."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # Should not happen in valid miconic states
             return float('inf') # Cannot make progress

        current_floor_idx = self.floor_map.get(current_lift_floor)
        if current_floor_idx is None:
             # Current lift floor not in floor map (e.g., no above facts, or invalid state)
             return float('inf')

        # Identify unserved passengers relevant to the goal
        unserved_passengers = [p for p in self.goal_passengers if f"(served {p})" not in state]

        if not unserved_passengers:
             # This should imply the goal is reached, which is checked at the start.
             # If we reach here, it means goal_passengers is empty or all are served,
             # but self.goals <= state was false. This shouldn't happen in miconic.
             # Return 0 as all relevant passengers are served.
             return 0

        required_stops_indices = set()
        passenger_action_cost = 0

        for p in unserved_passengers:
            p_origin = self.passenger_info.get(p, {}).get('origin')
            p_destin = self.passenger_info.get(p, {}).get('destin')

            # Ensure passenger info is available (should be from initial state)
            if p_origin is None or p_destin is None:
                 # Problem definition issue? Passenger in goal but no origin/destin in init?
                 return float('inf')

            is_boarded = f"(boarded {p})" in state
            is_at_origin = f"(origin {p} {p_origin})" in state

            if not is_boarded and is_at_origin:
                # Needs pickup
                if p_origin in self.floor_map:
                    required_stops_indices.add(self.floor_map[p_origin])
                    passenger_action_cost += 1 # Cost for board
                else:
                    # Origin floor not in map - problem definition issue?
                    return float('inf')

            if is_boarded:
                # Needs dropoff
                if p_destin in self.floor_map:
                    required_stops_indices.add(self.floor_map[p_destin])
                    passenger_action_cost += 1 # Cost for depart
                else:
                     # Destin floor not in map - problem definition issue?
                     return float('inf')

        # If no required stops (e.g., all unserved passengers are not at origin and not boarded - impossible in miconic)
        # or if all required stops are the current floor and no passenger actions needed (should be caught by goal check)
        # If unserved_passengers is not empty, there must be required stops in a valid state.
        # If required_stops_indices is unexpectedly empty when unserved_passengers is not, return infinity.
        if not required_stops_indices:
             # This case should ideally not be reached if unserved_passengers is not empty
             # and the state is valid according to miconic rules.
             # It might indicate a state where passengers are unserved but not at origin
             # and not boarded, which is impossible.
             # Let's return a base cost or infinity. Infinity is safer.
             # print("Error: unserved_passengers exists but no required stops found.")
             return float('inf')


        # Calculate movement cost
        min_stop_idx = min(required_stops_indices)
        max_stop_idx = max(required_stops_indices)

        # Movement cost: distance to nearest stop + span of stops
        dist_to_min = abs(current_floor_idx - min_stop_idx)
        dist_to_max = abs(current_floor_idx - max_stop_idx)

        movement_cost = min(dist_to_min, dist_to_max) + (max_stop_idx - min_stop_idx)


        total_heuristic = passenger_action_cost + movement_cost

        return total_heuristic
