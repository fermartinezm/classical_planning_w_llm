# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
# If running standalone or in a different environment, define a dummy base class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            self.task = task
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic __call__ method not implemented.")

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    # The number of parts must exactly match the number of args for a match
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    The heuristic estimates the remaining effort by summing:
    1. The number of unserved passengers (representing board/depart actions).
    2. The vertical distance spanning the current lift floor and all floors
       where unserved passengers need to be picked up or dropped off.

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < f3).
      This assumption is based on typical Miconic benchmarks and the structure
      of the `above` predicates in the provided examples. The numerical order
      defines the levels used for distance calculation.
    - The cost of moving between adjacent floors is 1. The heuristic estimates
      total vertical travel distance needed.
    - The cost of boarding or departing a passenger is 1.
    - The heuristic is non-admissible and aims to guide a greedy best-first search.

    # Heuristic Initialization
    - Extracts all floor objects and maps them to numerical levels based on their names.
    - Stores destination floors for all passengers from static facts.
    - Stores the set of goal literals (served passengers).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift by finding the fact `(lift-at ?f)`.
    2. Identify all passengers who are not yet served. These are the passengers `p`
       for whom the goal condition `(served p)` is not present in the current state.
    3. For these unserved passengers, identify the set of "required floors". A floor `f`
       is required if:
       - A passenger `p` is waiting at `f` (`(origin p f)` is true in the state).
       - A boarded passenger `p` needs to depart at `f` (`(boarded p)` is true in the state
         and `p`'s destination is `f`, obtained from static facts).
    4. Count the total number of unserved passengers (`N_unserved`). This count serves
       as a base estimate for the board and depart actions needed (at least one of each
       per passenger, possibly batched). If `N_unserved` is 0, the goal is reached,
       and the heuristic is 0.
    5. If there are unserved passengers (`N_unserved > 0`), calculate the estimated moves.
       - Get the numerical level for the current lift floor using the pre-computed
         `floor_to_level` map.
       - Get the numerical levels for all identified required floors.
       - Find the minimum (`min_l_inc`) and maximum (`max_l_inc`) levels among the
         current floor level and all required floor levels.
       - The estimated minimum number of moves is the difference between the maximum
         and minimum levels (`max_l_inc - min_l_inc`). This represents the total
         vertical span the lift must cover to visit all necessary floors, starting
         from its current position.
    6. The total heuristic value for the state is the sum of `N_unserved` and the
       estimated moves. This combines the cost of performing passenger transfers
       with the cost of lift movement.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels, passenger destinations,
        and goal conditions from the task definition.
        """
        self.task = task
        self.goals = task.goals

        # Extract all floor objects and map them to levels
        floor_names = set()
        # Look for floors in initial state and static facts
        for fact in task.initial_state | task.static:
            parts = get_parts(fact)
            # Floors appear as arguments in several predicates
            if parts and parts[0] in ['origin', 'destin', 'above', 'lift-at']:
                 # Arguments are passenger, floor or floor, floor
                 # Assume arguments starting with 'f' are floors based on examples.
                 # This is a domain-specific assumption based on typical naming conventions.
                 for part in parts[1:]:
                     if part.startswith('f'):
                         floor_names.add(part)

        # Sort floor names numerically (e.g., f1, f2, f10)
        # This assumes floor names are like 'f' followed by a number.
        try:
            sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # Fallback if floor names are not in f<number> format or empty
             # This simple heuristic might not work well then.
             # For robustness, could sort alphabetically or based on 'above' relations.
             # Sticking to the assumption for typical miconic problems.
             sorted_floor_names = sorted(list(floor_names)) # Simple alphabetical sort

        # Create floor name to level mapping (f1 -> 1, f2 -> 2, ...)
        self.floor_to_level = {f: i + 1 for i, f in enumerate(sorted_floor_names)}

        # Store passenger destinations from static facts
        self.passenger_destin = {}
        for fact in task.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destin[passenger] = floor

        # Store served goals (passengers who need to be served) for quick lookup
        self.served_goals = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state

        # 1. Identify current lift floor
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # If lift location is not found, state is likely invalid or terminal (goal)
        # In a goal state, N_unserved will be 0, handled below.
        # If it's an invalid non-goal state, return infinity.
        if current_floor is None:
             # This should not happen in valid states before goal
             return float('inf') # Indicate an unreachable or invalid state

        current_level = self.floor_to_level.get(current_floor)
        if current_level is None:
             # Should not happen if floor extraction is correct
             return float('inf') # Indicate an unknown floor


        # 2. Identify unserved passengers
        unserved_passengers = set()
        for goal_passenger in self.served_goals:
             if f"(served {goal_passenger})" not in state:
                  unserved_passengers.add(goal_passenger)

        # 4. Count unserved passengers
        n_unserved = len(unserved_passengers)

        # If all passengers are served, the heuristic is 0
        if n_unserved == 0:
            return 0

        # 3. Identify required floors for unserved passengers
        required_floors = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'origin' and parts[1] in unserved_passengers:
                # Passenger is waiting at origin - need to visit this floor for pickup
                required_floors.add(parts[2])
            elif parts[0] == 'boarded' and parts[1] in unserved_passengers:
                 # Passenger is boarded - need to visit their destination floor for dropoff
                 passenger = parts[1]
                 if passenger in self.passenger_destin:
                      required_floors.add(self.passenger_destin[passenger])
                 # else: Unserved boarded passenger with no destination in static? Invalid state.


        # If there are unserved passengers but no required floors, it implies
        # unserved passengers are neither waiting nor boarded. This should not
        # happen in a valid state. Return N_unserved as a minimal cost.
        if not required_floors:
             return n_unserved

        # 5. Calculate estimated moves
        required_levels = {self.floor_to_level.get(f) for f in required_floors}
        # Filter out any None values if a floor wasn't mapped (shouldn't happen)
        required_levels = {level for level in required_levels if level is not None}

        if not required_levels:
             # Again, should not happen if required_floors was non-empty and mapping is correct
             return n_unserved

        min_l = min(required_levels)
        max_l = max(required_levels)

        # Estimated moves: vertical span including current floor
        levels_including_current = required_levels | {current_level}
        min_l_inc = min(levels_including_current)
        max_l_inc = max(levels_including_current)
        estimated_moves = max_l_inc - min_l_inc

        # 6. Total heuristic value
        # Sum of board/depart actions (estimated by N_unserved) and estimated moves
        total_heuristic = n_unserved + estimated_moves

        return total_heuristic
