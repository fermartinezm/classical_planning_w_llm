from fnmatch import fnmatch
# Assuming a base Heuristic class is available as described in the problem
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided in the environment
# In a real scenario, this would be imported from the planning framework
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError("Subclasses must implement this method")

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions (board, depart, and moves)
    required to serve all passengers specified in the goal. It calculates the cost
    for each unserved passenger independently and sums these costs. The movement
    cost for a passenger is estimated as the sum of the distance from the lift's
    current floor to the passenger's origin (if waiting) and the distance from
    the origin to the destination.

    # Assumptions
    - All actions (board, depart, up, down) have a cost of 1.
    - The floors form a single vertical stack, and their order is defined by
      the `above` predicates.
    - Passengers are either waiting at their origin, boarded in the lift, or served.
    - The goal is always to have a specific set of passengers served.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Builds a mapping from floor names to numerical levels based on the `above`
      predicates. This mapping is used to calculate vertical distances (move costs).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift from the state. Determine its numerical level.
    2. Identify all passengers whose `(served ?p)` goal condition is not met in the current state. These are the unserved passengers we need to consider.
    3. Initialize the total heuristic cost to 0.
    4. For each unserved passenger:
       a. Get the passenger's destination floor and its numerical level (pre-calculated in `__init__`).
       b. Check if the passenger is currently waiting at their origin floor (i.e., `(origin ?p ?f)` is in the state).
          - If yes: Get the origin floor and its numerical level. The estimated cost for this passenger is:
            `abs(current_lift_level - origin_level) + 1 (board) + abs(origin_level - destination_level) + 1 (depart)`
          - If no (the passenger must be boarded, assuming a solvable state): The estimated cost for this passenger is:
            `abs(current_lift_level - destination_level) + 1 (depart)`
       c. Add the estimated cost for this passenger to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        - `self.destinations`: Maps passenger names to their destination floor names.
        - `self.floor_levels`: Maps floor names to their numerical level (starting from 1 for the lowest floor).
        - `self.goal_passengers`: Set of passengers that need to be served according to the goal.
        """
        super().__init__(task)

        # Extract destination floors for each passenger from static facts
        self.destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Build floor level mapping from "above" predicates
        # (above f_higher f_lower) means f_higher is immediately above f_lower
        above_map = {} # Maps f_lower -> f_higher
        all_floors = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                above_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Find the lowest floor (a floor that is not the 'f_higher' in any 'above' fact)
        # In a single stack, there should be exactly one such floor.
        highest_floors = set(above_map.values())
        lowest_floor = None
        for floor in all_floors:
            if floor not in highest_floors:
                 # Check if it's the lowest floor (not appearing as the second arg in any above)
                 # More robust way: find floor not appearing as first arg in any above
                 # Let's build the inverse map: f_higher -> f_lower
                 below_map = {f_higher: f_lower for f_lower, f_higher in above_map.items()}
                 # Lowest floor is one that is not a value in below_map
                 lower_floors = set(below_map.values())
                 lowest_candidates = [f for f in all_floors if f not in lower_floors]
                 if len(lowest_candidates) == 1:
                     lowest_floor = lowest_candidates[0]
                     break
                 # If multiple lowest candidates or none, the structure is unexpected.
                 # For typical miconic, there's one lowest floor.
                 # Let's assume the first candidate found is the lowest if multiple exist
                 # or handle error if none.
                 if lowest_candidates:
                     lowest_floor = lowest_candidates[0]
                     break


        self.floor_levels = {}
        if lowest_floor:
            current_floor = lowest_floor
            level = 1
            # Build map from f_lower -> f_higher again to traverse upwards
            above_map_traverse = {f_lower: f_higher for f_lower, f_higher in above_map.items()}
            
            # Build floor levels by traversing upwards from the lowest floor
            # Need to handle the case where above_map_traverse is empty (only one floor)
            if not above_map_traverse and len(all_floors) == 1:
                 self.floor_levels[list(all_floors)[0]] = 1
            else:
                current = lowest_floor
                level = 1
                while current in above_map_traverse or level == 1: # Keep going as long as there's a floor above, or if it's the first floor
                     self.floor_levels[current] = level
                     if current in above_map_traverse:
                         current = above_map_traverse[current]
                         level += 1
                     else:
                         break # Reached the highest floor


        # Identify passengers whose service is a goal
        self.goal_passengers = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                _, passenger = get_parts(goal)
                self.goal_passengers.add(passenger)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all goal passengers.
        """
        state = node.state

        # Find the current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state, but handle defensively
             # If lift location is unknown, cannot estimate moves. Return infinity or a large value.
             # For this heuristic, we assume a valid state with lift location.
             # If the state is a goal state, current_lift_floor might not matter if all served.
             # Let's check if it's a goal state first.
             if self.task.goal_reached(state):
                 return 0
             # If not goal and lift location is missing, something is wrong.
             # Return a high value to discourage this state.
             return float('inf') # Or some large number

        current_level = self.floor_levels.get(current_lift_floor)
        if current_level is None:
             # Should not happen if floor_levels is built correctly and state is valid
             return float('inf') # Or some large number


        total_heuristic = 0

        # Iterate through passengers that need to be served
        for passenger in self.goal_passengers:
            # Check if the passenger is already served
            if f"(served {passenger})" in state:
                continue # This passenger is served, no cost

            # Passenger is not served. Find their status (waiting or boarded)
            is_waiting = False
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    _, _, origin_floor = get_parts(fact)
                    is_waiting = True
                    break

            is_boarded = f"(boarded {passenger})" in state

            # Get destination floor and level
            destin_floor = self.destinations.get(passenger)
            if destin_floor is None:
                 # Should not happen if destinations are in static facts for goal passengers
                 # Handle defensively - this passenger cannot be served if destination is unknown
                 return float('inf') # Or some large number

            destin_level = self.floor_levels.get(destin_floor)
            if destin_level is None:
                 # Should not happen if floor_levels includes all destination floors
                 return float('inf') # Or some large number


            # Calculate individual cost based on status
            if is_waiting:
                # Passenger is waiting at origin
                origin_level = self.floor_levels.get(origin_floor)
                if origin_level is None:
                     # Should not happen
                     return float('inf') # Or some large number

                # Cost: Move to origin + Board + Move to destin + Depart
                # Simplified move cost: sum of absolute differences
                move_to_origin_cost = abs(current_level - origin_level)
                move_to_destin_cost = abs(origin_level - destin_level)
                board_cost = 1
                depart_cost = 1
                total_heuristic += move_to_origin_cost + board_cost + move_to_destin_cost + depart_cost

            elif is_boarded:
                # Passenger is boarded
                # Cost: Move to destin + Depart
                move_to_destin_cost = abs(current_level - destin_level)
                depart_cost = 1
                total_heuristic += move_to_destin_cost + depart_cost

            # Note: If a passenger is neither waiting nor boarded and not served,
            # something is wrong with the state representation or domain.
            # The current logic implicitly handles this by not adding cost if
            # is_waiting and is_boarded are both False (after checking if served).
            # This is correct as per domain rules (passengers are created at origin).

        return total_heuristic

