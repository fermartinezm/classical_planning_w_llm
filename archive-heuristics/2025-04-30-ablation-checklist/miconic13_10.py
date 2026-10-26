import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It counts the number of board and depart actions needed
    for unserved passengers and adds an estimate of the vertical movement
    cost for the lift.

    # Assumptions
    - Floors are linearly ordered (e.g., f1, f2, f3, ...). The `above`
      predicates in the static facts define this order, and we assume
      f_i is immediately above f_{i-1}.
    - The cost of each action (board, depart, up, down) is 1.

    # Heuristic Initialization
    - Extracts the floor names and creates a mapping from floor name to
      its numerical level based on the assumed linear order.
    - Extracts the destination floor for each passenger from static facts
      or initial state.
    - Stores the goal conditions (which passengers need to be served).

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current floor of the lift.
    2. Identify all passengers who are not yet served (by checking against goal conditions).
    3. For each unserved passenger:
       - If the passenger is waiting at their origin floor: They need a `board` action. The lift must visit their origin floor.
       - If the passenger is boarded: They need a `depart` action. The lift must visit their destination floor.
    4. Count the total number of `board` actions needed (one for each unserved passenger waiting at origin).
    5. Count the total number of `depart` actions needed (one for each unserved passenger who is boarded).
    6. Determine the set of *required floors* the lift must visit:
       - The current lift floor.
       - The origin floor for every unserved passenger waiting at their origin.
       - The destination floor for every unserved passenger who is boarded.
    7. Map these required floors to their numerical levels using the precomputed floor-to-level map.
    8. Calculate the movement cost: This is estimated as the difference between the maximum and minimum level among the required floors. This represents the minimum vertical distance the lift must span. If there are no required floors other than the current one, the movement cost is 0.
    9. The total heuristic value is the sum of the board actions needed, the depart actions needed, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        # Store goal conditions to identify unserved passengers.
        self.goals = task.goals

        # Extract floor names and create floor-to-level mapping.
        # Assumes floors are named f1, f2, ... and are linearly ordered.
        floor_names = set()
        # Floors are mentioned in 'above' predicates in static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'above':
                if len(parts) > 1:
                    floor_names.add(parts[1])
                if len(parts) > 2:
                    floor_names.add(parts[2])

        # Sort floor names numerically (e.g., f1, f2, f10)
        # Use a regex to extract the number part for sorting
        def sort_key(floor_name):
            match = re.match(r'f(\d+)', floor_name)
            if match:
                return int(match.group(1))
            return float('inf') # Put non-standard names at the end

        sorted_floor_names = sorted(list(floor_names), key=sort_key)

        # Create the floor name to level mapping (1-based index)
        self.floor_to_level = {f: i + 1 for i, f in enumerate(sorted_floor_names)}

        # Extract passenger destinations (static facts)
        self.passenger_destin = {}
        # Destinations are typically in static facts or initial state
        for fact in task.static | task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'destin':
                 if len(parts) == 3:
                     self.passenger_destin[parts[1]] = parts[2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break # Assuming only one lift-at fact

        if current_lift_floor is None:
             # Should not happen in a valid miconic state, but handle defensively
             # If lift location is unknown, cannot estimate movement.
             # Fallback to a simple relaxed heuristic? Or return infinity?
             # Let's assume valid states always have lift-at.
             # print("Warning: lift-at predicate not found in state.")
             return float('inf') # Cannot proceed without lift location

        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        # Goal passengers are those mentioned in the goal served predicates
        goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}
        unserved_passengers = {p for p in goal_passengers if p not in served_passengers}

        # If all goal passengers are served, the heuristic is 0
        if not unserved_passengers:
            return 0

        # 3. & 4. & 5. Count actions needed and identify required floors
        required_floors = {current_lift_floor} # Lift must be at its current floor
        actions_needed = 0 # Counts board and depart actions

        # Track current state of unserved passengers
        waiting_passengers_state = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*")}
        boarded_passengers_state = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        for p in unserved_passengers:
            # Check if passenger is waiting at origin
            if p in waiting_passengers_state:
                f_orig = waiting_passengers_state[p]
                required_floors.add(f_orig)
                actions_needed += 1 # Need 1 board action for this passenger

            # Check if passenger is boarded (and not yet served)
            elif p in boarded_passengers_state:
                 # Passenger is boarded, needs to depart at destination
                 f_dest = self.passenger_destin.get(p)
                 if f_dest: # Ensure destination is known
                     required_floors.add(f_dest)
                     actions_needed += 1 # Need 1 depart action for this passenger
                 # else: print(f"Warning: Destination for boarded passenger {p} not found.")


        # 7. Map required floors to levels
        required_levels = [self.floor_to_level.get(f) for f in required_floors if f in self.floor_to_level]

        # 8. Calculate movement cost
        movement_cost = 0
        if required_levels: # Only calculate if there are floors to visit
            min_level = min(required_levels)
            max_level = max(required_levels)
            movement_cost = max_level - min_level

        # 9. Total heuristic value
        total_cost = movement_cost + actions_needed

        return total_cost

