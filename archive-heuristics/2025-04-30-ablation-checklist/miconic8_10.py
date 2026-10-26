import fnmatch
from heuristics.heuristic_base import Heuristic
import re # Import re for numerical sorting of floor names

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential extra spaces within the fact string
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It counts the remaining board and depart actions needed
    and adds an estimate of the lift movement required to visit all
    relevant floors (origins of waiting passengers and destinations of
    boarded passengers).

    # Assumptions
    - Each unserved passenger needs one 'board' and one 'depart' action.
    - Lift movement is estimated based on the range of floors that need
      to be visited (origins and destinations of unserved passengers).

    # Heuristic Initialization
    - Determine the order of floors by parsing the 'above' predicates.
    - Map floor names (e.g., 'f1', 'f20') to numerical indices.
    - Store passenger destinations for quick lookup.
    - Identify all passenger objects.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1.  **Precomputation (__init__):**
        *   Parse `task.static` to build a mapping from floor names (like 'f1') to numerical indices (like 0, 1, 2...). The `(above f_i f_j)` facts imply `f_i` is above `f_j`. We can sort floors based on their numerical suffix.
        *   Parse `task.static` or `task.goals` to build a dictionary mapping each passenger to their destination floor object.
        *   Identify all passenger objects from `task.static` (e.g., from `origin` or `destin` facts).

    2.  **Heuristic Computation (__call__):**
        *   Get the current state (`node.state`).
        *   Find the current floor of the lift.
        *   Initialize counters for required actions: `board_actions = 0`, `depart_actions = 0`.
        *   Initialize a set to store the numerical indices of floors the lift *must* visit: `required_floor_nums = set()`.
        *   Iterate through all identified passenger objects:
            *   Check if the passenger is `(served ?p)` in the current state. If yes, this passenger is done; continue to the next.
            *   If the passenger is *not* served:
                *   Check if the passenger is at their `(origin ?p ?f_o)` in the current state. If yes:
                    *   Increment `board_actions` (one board action is needed).
                    *   Add the numerical index of `f_o` to `required_floor_nums`.
                *   Check if the passenger is `(boarded ?p)` in the current state. If yes:
                    *   Increment `depart_actions` (one depart action is needed).
                    *   Look up the passenger's destination floor `f_d` using the precomputed map.
                    *   Add the numerical index of `f_d` to `required_floor_nums`.
        *   Calculate the base heuristic cost from actions: `base_cost = board_actions + depart_actions`.
        *   Calculate the estimated lift movement cost:
            *   Find the numerical index of the current lift floor.
            *   If `required_floor_nums` is empty (all relevant passengers are already at the lift's current floor or all served), the movement cost is 0.
            *   Otherwise, find the minimum (`min_req_num`) and maximum (`max_req_num`) floor numbers in `required_floor_nums`.
            *   The estimated movement cost is the distance from the current lift floor to the lowest required floor, plus the distance from the lowest required floor to the highest required floor. This is `abs(current_lift_floor_num - min_req_num) + (max_req_num - min_req_num)`. This non-admissible estimate encourages moving towards the range of required floors and covering that range.
        *   The total heuristic value is `base_cost + movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and passenger objects from the task's static facts and goals.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Determine floor order and map floor names to numbers
        floor_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                floor_names.add(parts[1])
                floor_names.add(parts[2])
        # Also get floors from initial lift position if not in above facts
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "lift-at":
                 floor_names.add(parts[1])

        # Sort floors numerically based on the number part of the name (e.g., f1, f2, f10)
        # Use a regex to extract the number
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(re.search(r'\d+', f).group()))
        self.floor_to_num = {floor: i for i, floor in enumerate(sorted_floor_names)}
        self.num_to_floor = {i: floor for floor, i in self.floor_to_num.items()} # Useful for debugging

        # 2. Store passenger destinations
        self.passenger_to_destin_floor = {}
        # Destinations are typically in static facts or goals
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.passenger_to_destin_floor[passenger] = floor
        # Ensure goals are also covered, although destin is usually static
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 # We need the destination for unserved passengers.
                 # The destin fact is static, so the loop above should find it.
                 # This goal loop just confirms served passengers.
                 pass # Destinations are not in served goals

        # 3. Identify all passenger objects
        self.passengers = set(self.passenger_to_destin_floor.keys())
        # Also get passengers from initial origins if any are not in destin facts
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "origin":
                 self.passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor_obj = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor_obj = get_parts(fact)[1]
                break

        if current_lift_floor_obj is None:
             # This should not happen in a valid miconic state, but handle defensively
             # If lift location is unknown, cannot estimate movement. Return a large value?
             # Or maybe just count actions? Let's assume lift-at is always present.
             # For this heuristic, we must know the lift location.
             # If lift-at is missing, something is fundamentally wrong with the state.
             # Returning infinity or a very large number is safer for search.
             # However, the problem description implies valid states.
             # Let's assume lift_at is always found.
             pass # Assertion/Error handling could go here in a robust system

        current_lift_floor_num = self.floor_to_num[current_lift_floor_obj]

        unserved_origin_count = 0
        unserved_boarded_count = 0
        required_floor_nums = set() # Floors the lift MUST visit

        # Check each passenger's status
        for passenger in self.passengers:
            is_served = f"(served {passenger})" in state

            if not is_served:
                # Check if passenger is at origin
                at_origin = False
                origin_floor_obj = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        at_origin = True
                        origin_floor_obj = get_parts(fact)[2]
                        break

                # Check if passenger is boarded
                is_boarded = f"(boarded {passenger})" in state

                if at_origin:
                    unserved_origin_count += 1
                    if origin_floor_obj: # Should always be true if at_origin is true
                         required_floor_nums.add(self.floor_to_num[origin_floor_obj])
                elif is_boarded:
                    unserved_boarded_count += 1
                    # Add destination floor to required floors
                    destin_floor_obj = self.passenger_to_destin_floor.get(passenger)
                    if destin_floor_obj: # Should always be true for a valid problem
                         required_floor_nums.add(self.floor_to_num[destin_floor_obj])
                # else: passenger is not served, not at origin, not boarded - invalid state?
                # Or maybe they were just dropped off but served predicate not added yet?
                # The domain definition says depart adds served and removes boarded.
                # So unserved passengers are either at origin or boarded.

        # Calculate base action cost (board + depart actions)
        base_action_cost = unserved_origin_count + unserved_boarded_count

        # Calculate estimated lift movement cost
        movement_cost = 0
        if required_floor_nums:
            min_req_num = min(required_floor_nums)
            max_req_num = max(required_floor_nums)

            # Estimate movement: distance to the lowest required floor + distance to cover the range
            # This is a non-admissible estimate that encourages moving towards the "action zone"
            movement_cost = abs(current_lift_floor_num - min_req_num) + (max_req_num - min_req_num)

            # Alternative simpler estimate: distance to the closest required floor + distance to cover the range
            # This might be slightly better if the lift is far outside the range
            # closest_dist = min(abs(current_lift_floor_num - req_num) for req_num in required_floor_nums)
            # movement_cost = closest_dist + (max_req_num - min_req_num)
            # Let's stick to the first one for simplicity and reasonable performance.

            # If the current floor is one of the required floors, the distance to min_req_num might be 0
            # if current_lift_floor_num == min_req_num. The formula handles this.
            # If current_lift_floor_num is within the range [min_req_num, max_req_num],
            # abs(current_lift_floor_num - min_req_num) is the distance to the bottom,
            # and (max_req_num - min_req_num) covers the whole range. This seems reasonable.


        # Total heuristic is the sum of required actions and estimated movement
        total_cost = base_action_cost + movement_cost

        return total_cost

