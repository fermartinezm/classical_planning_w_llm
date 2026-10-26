from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort by summing two components:
    1. The number of remaining tasks for each unserved passenger (2 tasks if waiting at origin: board + depart; 1 task if boarded: depart).
    2. An estimate of the lift movement cost required to visit all necessary floors (origins for waiting passengers, destinations for boarded passengers).

    # Assumptions
    - Floors are named 'f<number>' (e.g., f1, f2, f10), and the number indicates the relative height.
    - The 'above' predicates define a linear order of floors where f<i> is above f<j> if i > j.
    - The lift can carry multiple passengers.
    - Actions (board, depart, up, down) have a cost of 1.

    # Heuristic Initialization
    - Parses static 'above' facts and initial state facts to identify all floors and determine their order, creating a mapping from floor name to an integer index. This allows calculating distances between floors.
    - Parses initial state 'destin' facts to store the destination floor for each passenger.
    - Parses initial state and goal facts to identify all passengers in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Initialize the total heuristic cost to 0.
    3. Initialize sets for floors the lift needs to visit: `floors_to_visit`.
    4. Iterate through all passengers identified during initialization:
       - Check if the passenger is already 'served' in the current state. If yes, they contribute 0 to the heuristic.
       - If not 'served':
         - Check if the passenger is waiting at their 'origin' floor in the current state. If yes:
           - Add 2 to the total heuristic cost (representing the future 'board' and 'depart' actions for this passenger).
           - Find the origin floor and add it to `floors_to_visit`.
         - Check if the passenger is 'boarded' in the current state. If yes:
           - Add 1 to the total heuristic cost (representing the future 'depart' action for this passenger).
           - Find the passenger's destination floor (stored during initialization) and add it to `floors_to_visit`.
    5. If all passengers are served (total_heuristic is 0 at this point), return 0.
    6. Calculate the estimated lift movement cost:
       - If `floors_to_visit` is empty, lift_cost is 0.
       - Otherwise, get the floor indices for the current lift floor and all floors in `floors_to_visit`.
       - Find the minimum and maximum floor indices among the needed floors.
       - The estimated lift movement cost is the minimum distance from the current lift floor's index to either the minimum or maximum needed floor index, plus the total range of needed floor indices. This estimates the cost of traveling to one end of the required range and sweeping through it.
    7. Add the estimated lift movement cost to the total heuristic cost.
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the list of all passengers.
        """
        self.task = task
        static_facts = task.static
        initial_state = task.initial_state
        goals = task.goals

        # 1. Determine floor order and create index map
        # Assuming floor names are f<number> and (above fi fj) means i > j
        all_floors = set()
        # Collect floors from static facts (especially 'above')
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                all_floors.add(parts[1])
                all_floors.add(parts[2])
        # Collect floors from initial state (lift-at, origin, destin)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ["lift-at", "origin", "destin"]:
                 if len(parts) > 1: # lift-at has 1 arg, origin/destin have 2
                     all_floors.add(parts[-1]) # The floor is the last argument

        # Extract numbers from floor names
        floor_numbers = {int(f[1:]) for f in all_floors if f.startswith('f') and f[1:].isdigit()}

        if not floor_numbers:
             # This case should ideally not happen in a valid miconic problem with floors
             # If it does, we can't calculate floor distances. Return a default or error.
             # For standard miconic, this won't be an issue.
             # Let's create a dummy index map if no floors found, though heuristic won't work well.
             if not all_floors:
                 self.floor_indices = {}
                 max_floor_number = 0 # Dummy value
             else:
                 # Handle floors without f<number> format if necessary, or raise error
                 raise ValueError(f"Could not parse floor names (expected f<number>) from {all_floors}")
        else:
            max_floor_number = max(floor_numbers)
            # Map f<i> to index (max_floor_number - i) so lowest floor has index 0
            self.floor_indices = {f'f{i}': max_floor_number - i for i in floor_numbers}


        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger, destination_floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = destination_floor

        # 3. Identify all passengers
        self.all_passengers = set()
        # Passengers are mentioned in origin, destin, boarded, served facts
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ["origin", "destin", "boarded"]:
                 if len(parts) > 1: # origin, destin, boarded have passenger as first arg
                    self.all_passengers.add(parts[1])
        for goal in goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served":
                 if len(parts) > 1: # served has passenger as first arg
                    self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        total_heuristic = 0
        floors_to_visit = set() # Floors the lift needs to go to (origins or destinations)

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # If lift location is unknown or no floors were parsed, return infinity
        if current_lift_floor is None or not self.floor_indices:
             return float('inf') # Should not be reachable in a solvable problem with floors

        # Calculate action cost and identify needed floors for unserved passengers
        for passenger in self.all_passengers:
            is_served = f'(served {passenger})' in state

            if not is_served:
                is_boarded = f'(boarded {passenger})' in state
                is_waiting = False
                origin_floor = None

                # Check if waiting at origin
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == "origin" and parts[1] == passenger:
                        is_waiting = True
                        origin_floor = parts[2]
                        break

                if is_waiting:
                    # Passenger needs board (1 action) and depart (1 action)
                    total_heuristic += 2
                    # Add origin floor to needed floors
                    if origin_floor in self.floor_indices:
                         floors_to_visit.add(origin_floor)
                    else:
                         # Origin floor not found in parsed floors - problem definition issue?
                         return float('inf') # Problematic state

                elif is_boarded:
                    # Passenger needs depart (1 action)
                    total_heuristic += 1
                    # Add destination floor to needed floors
                    if passenger in self.passenger_destinations:
                         dest_floor = self.passenger_destinations[passenger]
                         if dest_floor in self.floor_indices:
                            floors_to_visit.add(dest_floor)
                         else:
                             # Destination floor not found in parsed floors - problem definition issue?
                             return float('inf') # Problematic state
                    else:
                         # Destination not found for boarded passenger - problem definition issue?
                         return float('inf') # Problematic state

        # If total_heuristic is 0, it means all passengers are served (base case)
        if total_heuristic == 0:
             return 0

        # Calculate lift movement cost
        if not floors_to_visit:
            # This case should only happen if total_heuristic > 0 but floors_to_visit is empty,
            # which implies a problem definition issue (e.g., passenger needs service but no origin/destin floor)
            # Or perhaps a state where a passenger is waiting/boarded but their floor/dest is not among parsed floors.
            # Given the logic above, we should have returned inf already if floors were missing.
            # If we reach here and floors_to_visit is empty, it's likely a state where passengers need service
            # but their location/destination is the current lift floor, and no other floors are needed.
            # In this specific case, lift cost is 0.
            lift_cost = 0
        else:
            # Ensure current_lift_floor is in floor_indices (checked earlier, but defensive)
            if current_lift_floor not in self.floor_indices:
                 return float('inf') # Should not happen

            current_idx = self.floor_indices[current_lift_floor]
            needed_indices = [self.floor_indices[f] for f in floors_to_visit]
            min_needed_idx = min(needed_indices)
            max_needed_idx = max(needed_indices)

            # Cost to reach one end of the needed range + cost to traverse the range
            cost_to_min = abs(current_idx - min_needed_idx)
            cost_to_max = abs(current_idx - max_needed_idx)
            range_cost = max_needed_idx - min_needed_idx

            lift_cost = min(cost_to_min, cost_to_max) + range_cost

        total_heuristic += lift_cost

        return total_heuristic

