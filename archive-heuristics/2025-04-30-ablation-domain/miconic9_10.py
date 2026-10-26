from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
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
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the necessary 'board' actions, 'depart' actions, and estimates
    the minimum number of 'move' actions required to visit all relevant floors.

    # Assumptions
    - Floors are named 'fX' where X is an integer.
    - The 'above' predicate defines a linear order of floors, such that if
      '(above f_higher f_lower)' is true, f_higher is one level immediately
      above f_lower. This implies floors are ordered fN (lowest), f(N-1), ..., f1 (highest),
      where N is the total number of floors. The heuristic uses this order
      to calculate floor indices and distances.
    - Each 'move-up' or 'move-down' action changes the floor level by exactly one.
    - Each 'board' and 'depart' action costs 1.

    # Heuristic Initialization
    - Parses all floor objects to determine the total number of floors (N).
    - Creates a mapping from floor name ('fX') to its integer index (0-based),
      assuming fN is index 0, f(N-1) is index 1, ..., f1 is index N-1.
    - Parses all passenger objects.
    - Creates a mapping from passenger name to their destination floor name
      using the static 'destin' facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Identify all passengers and their current status: waiting at origin, boarded, or served.
    3. Count the number of passengers who are currently waiting at their origin floor.
       Each such passenger requires a 'board' action.
    4. Count the number of passengers who are currently boarded in the lift.
       Each such passenger requires a 'depart' action at their destination.
    5. Identify the set of floors the lift *must* visit to make progress on unserved passengers:
       - Origin floors of all waiting passengers.
       - Destination floors of all boarded passengers.
    6. Estimate the minimum number of 'move' actions required to visit all identified floors, starting from the current lift floor.
       - If there are no floors to visit, the move cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the required floors.
       - The estimated moves are the distance required to reach the nearest end of the required floor range from the current floor, plus the distance to traverse the entire required floor range. This is calculated as `(max_target_idx - min_target_idx) + min(abs(current_idx - min_target_idx), abs(current_idx - max_target_idx))`.
    7. The total heuristic value is the sum of:
       - The number of waiting passengers (representing required 'board' actions).
       - The number of boarded passengers (representing required 'depart' actions).
       - The estimated number of 'move' actions.
    8. If all passengers are served (goal state), the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about floors
        and passenger destinations.
        """
        self.goals = task.goals # Store goals, although not directly used in calculation
        static_facts = task.static

        # 1. Parse floors and build index map
        # Find all floor objects. Assuming floors are named f1, f2, ...
        floor_names = sorted([
            get_parts(fact)[1] for fact in task.facts if match(fact, "floor", "*")
        ], key=lambda f: int(f[1:])) # Sort by the number in the name

        self.num_floors = len(floor_names)
        # Assuming fN is lowest (index 0), f(N-1) is index 1, ..., f1 is highest (index N-1)
        # based on the common structure where (above fi fj) means i > j
        # and f1 is the highest floor. Let's re-verify this with example 1: (above f1 f2).
        # If f1 is index 1, f2 is index 0, then 1 > 0 holds.
        # Example 2: (above f1 f2), (above f2 f3), ..., (above f19 f20).
        # If f20=0, f19=1, ..., f1=19. Then index(f_higher) > index(f_lower) holds.
        # So, fX maps to index (N - X).
        self.floor_to_index = {f: self.num_floors - int(f[1:]) for f in floor_names}
        self.index_to_floor = {idx: f for f, idx in self.floor_to_index.items()}

        # 2. Parse passengers and their destinations
        self.passengers = [
            get_parts(fact)[1] for fact in task.facts if match(fact, "passenger", "*")
        ]
        self.passenger_destin = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1:]
                self.passenger_destin[p] = f

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Get current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        current_floor_idx = self.floor_to_index[current_lift_floor]

        # 2. Identify passenger statuses and relevant floors
        waiting_passengers_count = 0
        boarded_passengers_count = 0
        unserved_passengers_count = 0

        pickup_floors = set() # Origin floors of waiting passengers
        dropoff_floors = set() # Destination floors of boarded passengers

        passenger_status = {} # Map passenger to 'waiting', 'boarded', or 'served'
        passenger_origin = {} # Map passenger to origin floor if waiting

        for p in self.passengers:
            if f"(served {p})" in state:
                passenger_status[p] = 'served'
            elif f"(boarded {p})" in state:
                passenger_status[p] = 'boarded'
                boarded_passengers_count += 1
                unserved_passengers_count += 1
                # Add destination floor to dropoff_floors for boarded passengers
                if p in self.passenger_destin:
                     dropoff_floors.add(self.passenger_destin[p])
            else:
                # Must be waiting at origin
                passenger_status[p] = 'waiting'
                waiting_passengers_count += 1
                unserved_passengers_count += 1
                # Find origin floor
                for fact in state:
                    if match(fact, "origin", p, "*"):
                        origin_floor = get_parts(fact)[2]
                        passenger_origin[p] = origin_floor
                        pickup_floors.add(origin_floor)
                        break # Found origin for this passenger

        # If no unserved passengers, it should be a goal state (already checked)
        # If we reach here and unserved_passengers_count is 0, something is wrong
        # or the goal check above is sufficient.

        # 3. Count board and depart actions needed
        # Each waiting passenger needs 1 board action
        num_board_actions = waiting_passengers_count
        # Each boarded passenger needs 1 depart action
        num_depart_actions = boarded_passengers_count

        # 4. Estimate move actions
        all_target_floors = pickup_floors | dropoff_floors

        num_move_actions = 0
        if all_target_floors:
            target_indices = {self.floor_to_index[f] for f in all_target_floors}
            min_target_idx = min(target_indices)
            max_target_idx = max(target_indices)

            # Estimate moves to visit all target floors from current floor
            # This is the distance to the nearest end of the target range
            # plus the distance to traverse the entire target range.
            dist_to_min = abs(current_floor_idx - min_target_idx)
            dist_to_max = abs(current_floor_idx - max_target_idx)
            span = max_target_idx - min_target_idx

            # If current is outside the range [min_target_idx, max_target_idx]
            if current_floor_idx < min_target_idx:
                 # Must go up at least to max_target_idx
                 num_move_actions = max_target_idx - current_floor_idx
            elif current_floor_idx > max_target_idx:
                 # Must go down at least to min_target_idx
                 num_move_actions = current_floor_idx - min_target_idx
            else:
                 # Current is within the range. Must go to one end and sweep to the other.
                 # Option 1: Go down to min, then up to max. Moves = (current - min) + (max - min)
                 # Option 2: Go up to max, then down to min. Moves = (max - current) + (max - min)
                 # Minimum moves = (max - min) + min(current - min, max - current)
                 num_move_actions = span + min(dist_to_min, dist_to_max)


        # 5. Total heuristic value
        # Heuristic = (board actions) + (depart actions) + (move actions)
        heuristic_value = num_board_actions + num_depart_actions + num_move_actions

        return heuristic_value

