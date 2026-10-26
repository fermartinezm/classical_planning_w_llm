from fnmatch import fnmatch
import re
# Assuming the Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the input is treated as a string and remove leading/trailing whitespace
    fact_str = str(fact).strip()
    # Check if it looks like a PDDL fact string
    if fact_str.startswith('(') and fact_str.endswith(')'):
        # Remove parentheses and split by whitespace
        return fact_str[1:-1].split()
    # Return empty list for invalid fact strings
    return []

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def numerical_sort_key(floor_name):
    """
    Extract the numerical part from floor names like 'f1', 'f10' for sorting.
    This assumes floor names follow the pattern 'f' followed by digits.
    """
    match = re.match(r'f(\d+)', floor_name)
    if match:
        # Return the integer value for numerical sorting
        return int(match.group(1))
    # Return a large value for names that don't match the pattern,
    # placing them at the end (should not happen in standard miconic).
    return float('inf')

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the estimated non-movement actions (board and depart)
    and the estimated minimum lift movement actions needed to visit all
    required floors (passenger origins and destinations).

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < f10).
    - The lift can carry all boarded passengers simultaneously.
    - The lift must visit a floor to pick up a waiting passenger or drop off
      a boarded passenger.
    - The minimum movement cost to visit a set of floors is estimated by
      traveling from the current floor to the closer of the two extreme
      required floor levels (lowest or highest) and then sweeping across the
      entire range of required floors. This is a simplified estimate of the
      traveling salesman problem on a line.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from static facts.
    - Determines the ordering and numerical level for each floor based on
      the static `above` facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift by finding the `(lift-at ?f)` fact in the state.
    2. Identify all passengers who have not yet been served by checking the goal conditions
       and the `(served ?p)` facts in the state.
    3. If there are no unserved passengers, the goal is reached, and the heuristic value is 0.
    4. If there are unserved passengers, categorize them as either waiting at their origin
       floor (`(origin ?p ?f)` is true) or boarded (`(boarded ?p)` is true).
    5. Calculate the estimated non-movement cost:
       - Each waiting passenger requires one `board` action.
       - Each unserved passenger (both waiting and boarded) requires one `depart` action.
       - The non-movement cost is the sum of required board actions and required depart actions.
    6. Identify the set of floors that the lift *must* visit to serve the unserved passengers:
       - This includes the origin floor for every waiting passenger.
       - This includes the destination floor for every unserved passenger (both waiting and boarded).
    7. If the set of required floors to visit is empty (this should only happen if there are no unserved passengers, which is handled in step 3), the movement cost is 0.
    8. If the set of required floors is not empty, determine the minimum and maximum floor levels among these required floors using the pre-calculated floor level map.
    9. Calculate the estimated minimum lift movement cost:
       - Get the level of the current lift floor.
       - Calculate the distance needed to reach the lowest required floor level from the current level, plus the distance to sweep from the lowest to the highest required level.
       - Calculate the distance needed to reach the highest required floor level from the current level, plus the distance to sweep from the highest to the lowest required level (which is the same distance as sweeping up).
       - The minimum lift movement cost is the minimum of these two calculated distances. This estimates the cost of traveling to one extreme of the required floors and then traversing the entire range.
    10. The total heuristic value is the sum of the estimated non-movement cost (step 5) and the estimated minimum lift movement cost (step 9).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions.
        - Destination floor for each passenger from static facts.
        - Floor level mapping from above facts.
        """
        self.goals = task.goals

        self.destinations = {}
        floor_names_set = set()
        # Collect all passenger names that appear in goal or static destin facts
        all_passenger_names_set = set()

        # Parse static facts to get destinations and floor names
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "destin":
                # Fact is like (destin p1 f2)
                if len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    self.destinations[passenger] = floor
                    all_passenger_names_set.add(passenger)
            elif predicate == "above":
                # Fact is like (above f1 f2)
                if len(parts) == 3:
                    floor1, floor2 = parts[1], parts[2]
                    floor_names_set.add(floor1)
                    floor_names_set.add(floor2)

        # Also collect passenger names from goal facts (e.g., (served p1))
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts and parts[0] == "served" and len(parts) == 2:
                 all_passenger_names_set.add(parts[1])

        # Store all relevant passenger names
        self.all_passengers = all_passenger_names_set

        # Sort floor names numerically and create the level map
        sorted_floor_names = sorted(list(floor_names_set), key=numerical_sort_key)
        # Assign levels starting from 1
        self.floor_level_map = {floor: level for level, floor in enumerate(sorted_floor_names, 1)}

        # Store a set of served goal facts for quick lookup
        self.served_goals = {str(g) for g in self.goals if get_parts(str(g))[0] == 'served'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify the current floor of the lift.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown, heuristic is infinite (should not happen in valid states)
        if current_lift_floor is None:
             return float('inf')

        # 2. Identify all passengers who have not yet been served.
        # Check against all passengers identified during initialization
        unserved_passengers = {
            p for p in self.all_passengers if f"(served {p})" not in state
        }

        # 3. If there are no unserved passengers, the goal is reached.
        if not unserved_passengers:
            return 0

        # 4. Categorize unserved passengers. Count waiting.
        waiting_passengers = set()
        boarded_passengers = set() # Unserved and boarded
        # Need to find origin floors for waiting passengers from the current state
        origin_floors_in_state = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == "boarded" and len(parts) == 2:
                 p = parts[1]
                 if p in unserved_passengers:
                     boarded_passengers.add(p)
            elif predicate == "origin" and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 if p in unserved_passengers:
                     waiting_passengers.add(p)
                     origin_floors_in_state[p] = f

        # Ensure all unserved passengers are either waiting or boarded
        # assert len(unserved_passengers) == len(waiting_passengers) + len(boarded_passengers)

        # 5. Estimated non-movement cost
        # Each waiting passenger needs 1 board action.
        # Each unserved passenger needs 1 depart action.
        h_non_move = len(waiting_passengers) + len(unserved_passengers)

        # 6. Identify the set of floors that the lift must visit.
        floors_to_visit = set()

        # Origin floors of waiting passengers
        for p in waiting_passengers:
             origin_floor = origin_floors_in_state.get(p)
             if origin_floor: # Should always find one for a waiting passenger
                 floors_to_visit.add(origin_floor)

        # Destination floors of all unserved passengers
        for p in unserved_passengers:
            destin_floor = self.destinations.get(p)
            if destin_floor: # Should always find one if passenger is in self.destinations
                floors_to_visit.add(destin_floor)

        # If no floors to visit (e.g., all unserved passengers are already at their destination
        # but not served - this shouldn't happen in a valid state unless they are boarded
        # and the lift is at the destination, in which case depart is possible),
        # the move cost is 0. This case is implicitly handled below if floors_to_visit is empty.
        if not floors_to_visit:
             # This state implies unserved passengers exist but require no floor visits?
             # This could happen if all unserved passengers are boarded and the lift
             # is currently at the destination floor for *all* of them.
             # In this specific case, only depart actions are needed.
             # The non-movement cost already covers the depart actions.
             # The move cost should be 0.
             min_moves_for_lift = 0
        else:
            # 8. Calculate min and max floor levels among required floors.
            required_levels = [self.floor_level_map[f] for f in floors_to_visit if f in self.floor_level_map]
            if not required_levels: # Should not happen if floors_to_visit is not empty and floor_level_map is correct
                 min_moves_for_lift = 0
            else:
                min_req_level = min(required_levels)
                max_req_level = max(required_levels)
                current_lift_level = self.floor_level_map.get(current_lift_floor, float('inf')) # Handle unknown floor gracefully

                if current_lift_level == float('inf'):
                     return float('inf') # Unknown lift location

                # 9. Calculate estimated minimum lift movement cost.
                # Estimate the cost to reach one extreme of the required floors and sweep the range.
                # Option 1: Go to min_req_level, then sweep up to max_req_level.
                moves1 = abs(current_lift_level - min_req_level) + (max_req_level - min_req_level)
                # Option 2: Go to max_req_level, then sweep down to min_req_level.
                moves2 = abs(current_lift_level - max_req_level) + (max_req_level - min_req_level)

                min_moves_for_lift = min(moves1, moves2)

        # 10. Total heuristic value.
        total_cost = h_non_move + min_moves_for_lift

        return total_cost
