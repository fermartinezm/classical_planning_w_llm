from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

# Helper function to extract components from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a given pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved passengers.
    It sums the number of necessary board/depart actions for each unserved passenger
    and an estimate of the lift movement cost to visit all required floors (origins for waiting passengers,
    destinations for boarded passengers).

    # Assumptions
    - Floors are named f1, f2, f3, ... and are ordered numerically (f1 is the lowest, f2 is above f1, etc.).
    - The predicate `(above f_i f_j)` is true if floor f_i is numerically lower than floor f_j.
    - All actions (board, depart, up, down) have a cost of 1.
    - The heuristic is not required to be admissible.

    # Heuristic Initialization
    - Maps floor names (e.g., 'f1', 'f10') to their corresponding integer levels (1, 10). This mapping is crucial for calculating floor distances.
    - Stores the destination floor for each passenger by parsing the static facts.
    - Identifies the set of all passengers that need to be served based on the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is a goal state. If all goal conditions are met (all required passengers are served), the heuristic value is 0.
    2. Identify the current floor of the lift by finding the `(lift-at ?floor)` fact in the state. If the lift location cannot be determined (which should not happen in valid states), return infinity.
    3. Initialize counters for the number of waiting passengers and the number of boarded passengers among those who are not yet served.
    4. Initialize a set to store the floors the lift must visit (required stops). These include the origin floors of waiting passengers and the destination floors of boarded passengers.
    5. Iterate through the set of all passengers that need to be served (this set was identified during the heuristic's initialization from the goal facts):
       - For each passenger, check if they are already served by looking for the fact `(served passenger)` in the current state. If found, this passenger is done, and we move to the next passenger.
       - If the passenger is not served:
         - Check if the passenger is waiting at their origin floor. This is done by iterating through the state facts and looking for a fact that starts with `(origin passenger `. If such a fact is found:
           - Increment the waiting passenger count.
           - Extract the origin floor from this fact and add it to the set of required stops.
           - Mark that this passenger's location has been found (they are waiting).
         - If the passenger was not found waiting, check if they are boarded in the lift. This is done by checking if the fact `(boarded passenger)` is present in the current state. If it is:
           - Increment the boarded passenger count.
           - Look up the passenger's destination floor using the destination map created during initialization.
           - Add their destination floor to the set of required stops.
    6. Calculate the base action cost: This estimates the minimum number of board and depart actions required for the unserved passengers. Each waiting passenger will need to be boarded (1 action) and later departed (1 action), totaling 2 actions. Each boarded passenger will need to be departed (1 action). The base action cost is calculated as `(waiting passenger count * 2) + (boarded passenger count * 1)`.
    7. Calculate the estimated lift movement cost:
       - If the set of required stops is empty (meaning all unserved passengers are either already at their destination and boarded, or there are no unserved passengers requiring a specific stop), the estimated move cost is 0.
       - If there are required stops:
         - Get the integer level for the current lift floor using the floor-to-integer mapping created during initialization. If the current floor is not in the map (unexpected), return infinity.
         - Get the integer levels for all floors in the set of required stops. If any required floor is not in the map (unexpected), return infinity.
         - Find the minimum and maximum floor levels among the required stops.
         - The estimated move cost is calculated as the minimum distance from the current lift floor level to either the minimum required floor level or the maximum required floor level, plus the difference between the maximum and minimum required floor levels (which represents the span of floors the lift must traverse). This formula estimates the cost to travel to one end of the required floor range and then sweep across the entire range. The formula is: `min(abs(current_floor_idx - min_required_idx), abs(current_floor_idx - max_required_idx)) + (max_required_idx - min_required_idx)`.
    8. The total heuristic value for the state is the sum of the base action cost and the estimated lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor mapping, passenger destinations,
        and the set of passengers to be served from the task definition.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find all floor objects

        # 1. Map floor names to integer levels
        floor_names = set()
        # Collect potential floor names from all facts in initial state and static facts.
        # We assume floor names start with 'f' followed by digits.
        all_facts = self.initial_state | self.static
        for fact_str in all_facts:
             parts = get_parts(fact_str)
             for part in parts:
                 # Check if it starts with 'f' and the rest are digits
                 if part.startswith('f') and part[1:].isdigit():
                     floor_names.add(part)

        # Create the mapping 'fi' -> i
        self.floor_to_int = {}
        for floor_name in floor_names:
            # Extract integer suffix and use it as the floor level
            try:
                floor_level = int(floor_name[1:])
                self.floor_to_int[floor_name] = floor_level
            except (ValueError, IndexError):
                 # This should not happen with standard miconic floor names,
                 # but handle defensively if unexpected object names exist.
                 pass


        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact_str in self.static:
            # Use the match helper for parsing static facts like (destin passenger floor)
            if match(fact_str, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact_str)
                self.passenger_destinations[passenger] = destination_floor

        # 3. Identify passengers to serve
        self.passengers_to_serve = set()
        for goal_fact in self.goals:
            # Goal facts are typically simple predicates like (served p1)
            if goal_fact.startswith("(served "):
                 parts = get_parts(goal_fact)
                 # Ensure the fact has the expected structure (served passenger)
                 if len(parts) == 2:
                    _, passenger = parts
                    self.passengers_to_serve.add(passenger)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal state.

        Args:
            node: The current state node in the search tree.

        Returns:
            An integer estimate of the remaining actions, or float('inf') if the state is likely invalid.
        """
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify current lift floor
        current_lift_floor = None
        # Iterate through state facts to find the lift location predicate
        for fact_str in state:
            if fact_str.startswith("(lift-at "):
                parts = get_parts(fact_str)
                # Ensure the fact has the expected structure (lift-at floor)
                if len(parts) == 2:
                    _, current_lift_floor = parts
                    break # Found the lift location, stop searching

        # If the lift location is not found or the floor name is not recognized,
        # this state is likely invalid or unreachable in a standard problem.
        # Return infinity to prune this branch.
        if current_lift_floor is None or current_lift_floor not in self.floor_to_int:
             return float('inf')


        # 3, 4, 5. Identify unserved passengers, required stops, and counts
        required_stops = set()
        waiting_count = 0
        boarded_count = 0

        # Iterate through the passengers we know need to be served from the goal
        for passenger in self.passengers_to_serve:
            # Check if this passenger is already served in the current state
            if f"(served {passenger})" in state:
                continue # This passenger is already at their destination and departed

            # Check if the passenger is waiting at their origin floor
            is_waiting = False
            # Construct the expected prefix for the origin fact
            origin_fact_prefix = f"(origin {passenger} "
            # Iterate through state facts to find the origin location for this specific passenger
            for fact_str in state:
                 if fact_str.startswith(origin_fact_prefix):
                    parts = get_parts(fact_str)
                    # Ensure the fact has the expected structure (origin passenger floor)
                    if len(parts) == 3:
                        _, _, origin_floor = parts
                        waiting_count += 1
                        required_stops.add(origin_floor)
                        is_waiting = True
                        break # Found the origin fact for this passenger, move on

            # If the passenger was not found waiting, check if they are boarded in the lift
            if not is_waiting and f"(boarded {passenger})" in state:
                 boarded_count += 1
                 # Get the passenger's destination floor from the pre-calculated map
                 destination_floor = self.passenger_destinations.get(passenger)
                 # The destination should always exist for a passenger mentioned in the goal
                 if destination_floor:
                     required_stops.add(destination_floor)
                 # else: # Defensive: passenger boarded but no destination found? Invalid state?
                 #    pass # Could log a warning or return infinity if this indicates a problem


        # 6. Calculate base action cost
        # This estimates the minimum number of board and depart actions needed.
        # Each waiting passenger needs 1 board + 1 depart = 2 actions.
        # Each boarded passenger needs 1 depart = 1 action.
        action_cost = (waiting_count * 2) + boarded_count

        # 7. Calculate estimated lift movement cost
        move_cost = 0
        # Only calculate move cost if there are floors the lift needs to visit
        if required_stops:
            # Get the integer level for the current lift floor
            current_lift_floor_idx = self.floor_to_int[current_lift_floor]

            # Get the integer levels for all required stop floors
            required_stops_idx = set()
            for f in required_stops:
                 # Ensure all required stop floors were identified during initialization
                 if f not in self.floor_to_int:
                      # Unexpected floor name in required stops. Return infinity.
                      return float('inf')
                 required_stops_idx.add(self.floor_to_int[f])

            # Find the minimum and maximum floor levels among the required stops
            min_s_idx = min(required_stops_idx)
            max_s_idx = max(required_stops_idx)

            # Estimated moves = distance from current floor to the closest end of the required range
            #                 + the span of the required range.
            # This is min(abs(current_floor_idx - min_required_idx), abs(current_floor_idx - max_required_idx)) + (max_required_idx - min_required_idx)
            move_cost = min(abs(current_lift_floor_idx - min_s_idx), abs(current_lift_floor_idx - max_s_idx)) + (max_s_idx - min_s_idx)

        # 8. Total heuristic value is the sum of action cost and movement cost
        total_cost = action_cost + move_cost

        return total_cost
