import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potentially complex arguments like "(at obj (loc city))" if needed,
    # but for miconic, simple split is sufficient.
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(lift-at f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions required to serve all passengers.
    The heuristic sums:
    1. The number of 'board' actions needed (one for each waiting passenger).
    2. The number of 'depart' actions needed (one for each unserved passenger).
    3. An estimate of the 'up'/'down' movement actions needed to visit all
       required floors (origins of waiting passengers, destinations of boarded passengers).

    The movement cost is estimated as the distance between the lowest and highest
    required floors, plus the minimum distance from the current floor to either
    the lowest or highest required floor. This encourages moving towards the
    "range" of necessary stops.

    This heuristic is not admissible but aims to be informative for greedy search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor ordering (mapping floor names to numerical indices).
        - Passenger destinations.
        - List of all passengers.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Build floor ordering (name to index and index to name)
        # Assumes floor names are f1, f2, ..., fn and are ordered numerically.
        # Extract all unique floor names from initial state and static facts.
        all_floor_names = set()
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             for part in parts:
                 # Simple check: starts with 'f' and contains digits
                 if part.startswith('f') and any(char.isdigit() for char in part):
                     all_floor_names.add(part)

        # Sort floor names numerically based on the number suffix
        def floor_sort_key(floor_name):
            match = re.match(r'f(\d+)', floor_name)
            if match:
                return int(match.group(1))
            return float('inf') # Put non-standard names at the end

        sorted_floor_names = sorted(list(all_floor_names), key=floor_sort_key)

        self.floor_to_index = {name: i + 1 for i, name in enumerate(sorted_floor_names)}
        self.index_to_floor = {i + 1: name for i, name in enumerate(sorted_floor_names)}

        # 2. Store passenger destinations for quick lookup
        self.passenger_destin = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destin[passenger] = floor

        # 3. Get list of all passengers
        self.all_passengers = set()
        for fact in task.initial_state | task.static:
            parts = get_parts(fact)
            if parts and parts[0] in ["origin", "destin", "boarded", "served"]:
                 if len(parts) > 1: # Ensure there's a passenger argument
                     self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift floor
        current_floor_name = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor_name = get_parts(fact)[1]
                break

        if current_floor_name is None:
             # This should not happen in a valid miconic state, but handle defensively
             # If lift location is unknown, we can't estimate movement.
             # Fallback to a simple count of unserved passengers.
             unserved_count = sum(1 for p in self.all_passengers if f"(served {p})" not in state)
             # Each unserved needs at least board + depart
             return unserved_count * 2


        # Identify unserved passengers and required floors
        unserved_passengers = set()
        required_floors_names = set()
        h_board_depart = 0

        for p in self.all_passengers:
            if f"(served {p})" not in state:
                unserved_passengers.add(p)

                # Check if waiting at origin
                is_waiting = False
                for fact in state:
                    if match(fact, "origin", p, "*"):
                        origin_floor = get_parts(fact)[2]
                        required_floors_names.add(origin_floor)
                        h_board_depart += 1 # Need a 'board' action
                        is_waiting = True
                        break # Found origin, move to next passenger state check

                # Check if boarded (if not waiting)
                if not is_waiting:
                    if f"(boarded {p})" in state:
                        destin_floor = self.passenger_destin.get(p)
                        if destin_floor: # Should always have a destination
                            required_floors_names.add(destin_floor)
                        # No board action needed, already boarded
                    # else: passenger is served (already checked) or in an invalid state

                h_board_depart += 1 # Need a 'depart' action for any unserved passenger


        # Calculate movement cost
        move_cost = 0
        if required_floors_names:
            current_idx = self.floor_to_index.get(current_floor_name)
            if current_idx is None:
                 # Should not happen if floor names are consistent
                 # Fallback to simple board/depart count
                 return h_board_depart # Movement cost unknown

            required_indices = {self.floor_to_index[f] for f in required_floors_names if f in self.floor_to_index}

            if required_indices: # Ensure there are valid required floors
                min_idx = min(required_indices)
                max_idx = max(required_indices)

                # Estimate moves to cover the range of required floors
                # Distance to one end + distance from that end to the other end
                dist_to_min = abs(current_idx - min_idx)
                dist_to_max = abs(current_idx - max_idx)

                # Cost to reach one end and traverse the range
                # Option 1: Go to min_idx first, then traverse to max_idx
                cost1 = dist_to_min + (max_idx - min_idx)
                # Option 2: Go to max_idx first, then traverse to min_idx
                cost2 = dist_to_max + (max_idx - min_idx)

                # The minimum moves to visit all floors in the range [min_idx, max_idx]
                # starting from current_idx is min(cost1, cost2).
                # However, a simpler estimate that still captures range and current position:
                # Distance between min and max required floors + distance from current to the closer end.
                # This is (max_idx - min_idx) + min(dist_to_min, dist_to_max)
                # Let's use this simpler one as it's common and effective for greedy search.
                move_cost = (max_idx - min_idx) + min(dist_to_min, dist_to_max)

                # If the current floor is one of the required floors, we might save some moves.
                # If current_idx is min_idx or max_idx, min(dist_to_min, dist_to_max) is 0.
                # If current_idx is between min_idx and max_idx, min(dist_to_min, dist_to_max) is the distance to the closer end.
                # This formula seems reasonable.

        # Total heuristic is sum of board/depart actions and estimated movement actions
        return h_board_depart + move_cost

