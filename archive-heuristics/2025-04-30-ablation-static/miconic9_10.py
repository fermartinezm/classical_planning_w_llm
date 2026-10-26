from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided in the execution environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """Dummy Heuristic base class for standalone execution."""
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic not implemented")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    # Ensure the number of parts is at least the number of args for a potential match
    if len(parts) < len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers that are not yet served. It calculates the cost for each
    unserved passenger independently and sums these costs. For a passenger,
    the cost includes moving the lift to their origin floor (if waiting),
    boarding, moving the lift to their destination floor, and departing.

    # Assumptions
    - The lift has infinite capacity.
    - The floor structure is a single linear sequence defined by 'above' predicates.
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - Passenger destinations are static throughout the problem.

    # Heuristic Initialization
    - Extracts the floor ordering from the static 'above' facts to create a
      mapping from floor names to numerical levels. Assumes a linear floor structure
      or handles multiple towers by assigning levels within each.
    - Stores the destination floor for each passenger that needs to be served
      according to the task's goal conditions, reading destinations from the
      initial state's 'destin' facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Initialize the total heuristic cost to 0.
    3. Identify all passengers that are required to be served according
       to the task's goal conditions (identified during initialization).
    4. For each goal passenger:
       a. Check if the passenger is already 'served' in the current state. If yes,
          this passenger contributes 0 to the heuristic.
       b. If the passenger is 'boarded' in the current state:
          - Get the passenger's destination floor (stored during initialization).
          - Calculate the number of floor movements needed from the current lift
            floor to the passenger's destination floor:
            `abs(current_lift_level - destination_level)`.
          - Add the movement cost and the 'depart' action cost (1) to the total.
       c. If the passenger is 'waiting' at an origin floor `f_origin`
          (i.e., `(origin passenger f_origin)` is in the state):
          - Get the passenger's destination floor (stored during initialization).
          - Get the passenger's *current* origin floor from the state fact `(origin passenger f_origin)`.
          - Calculate the number of floor movements needed from the current lift
            floor to the passenger's current origin floor:
            `abs(current_lift_level - current_origin_level)`.
          - Add the movement cost and the 'board' action cost (1).
          - Calculate the number of floor movements needed from the passenger's
            current origin floor to their destination floor:
            `abs(current_origin_level - destination_level)`.
          - Add this movement cost and the 'depart' action cost (1) to the total.
    5. The total accumulated cost is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger info.
        """
        # Assuming task object has attributes: goals, static, initial_state
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Extract Floor Ordering and Levels ---
        floor_above_map = {}
        all_floors = set()
        floors_with_lower_neighbor = set() # Floors that appear as f_higher in (above f_higher f_lower)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_higher, f_lower = parts[1], parts[2]
                floor_above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                floors_with_lower_neighbor.add(f_higher)

        # Find the lowest floor(s) - those in all_floors but not in floors_with_lower_neighbor
        lowest_floors = list(all_floors - floors_with_lower_neighbor)

        self.floor_levels = {}

        if not all_floors:
             # No floors found
             pass # Warning already handled by returning inf in __call__ if lift-at is missing
        elif len(lowest_floors) == 1:
            # Standard case: single linear tower
            lowest_floor = lowest_floors[0]
            current_floor = lowest_floor
            level = 0
            while current_floor in all_floors:
                self.floor_levels[current_floor] = level
                level += 1
                current_floor = floor_above_map.get(current_floor)
                if not current_floor:
                    break # Reached the highest floor
        else:
             # Handle cases with multiple towers or disconnected floors.
             # Assign level 0 to all identified lowest floors and traverse upwards from each.
             # This assumes movement within a tower is possible, but movement between towers is not
             # (which is true for miconic). The heuristic will sum costs across towers.
             # print(f"Warning: Found {len(lowest_floors)} potential lowest floors: {lowest_floors}. Handling as multiple towers.")
             visited_floors = set()
             for start_floor in lowest_floors:
                 if start_floor in visited_floors:
                     continue # Already processed this tower

                 current_floor = start_floor
                 level = 0
                 # Traverse upwards from this lowest floor
                 while current_floor in all_floors and current_floor not in visited_floors:
                     self.floor_levels[current_floor] = level
                     visited_floors.add(current_floor)
                     level += 1
                     current_floor = floor_above_map.get(current_floor)
                     if not current_floor:
                         break # Reached the top of this tower

             # Add any floors mentioned in 'above' but not part of the main tower traversal
             # (e.g., isolated floors or complex structures not handled by simple traversal)
             # This might indicate a problem with the domain definition or instance.
             # For robustness, assign level 0 to any remaining unassigned floors.
             remaining_floors = all_floors - set(self.floor_levels.keys())
             if remaining_floors:
                 # print(f"Warning: Some floors were not included in level assignment: {remaining_floors}. Assigning level 0.")
                 for floor in remaining_floors:
                     self.floor_levels[floor] = 0


        # --- Extract Passenger Destinations ---
        self.passenger_destinations = {}
        # Destinations are static, read from initial state
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                p, f_destin = parts[1], parts[2]
                self.passenger_destinations[p] = f_destin

        # Identify all passengers that need to be served from the goal state
        self.goal_passengers = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == "served" and len(get_parts(g)) == 2}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings)

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state, but handle defensively
             # A state without a lift location is invalid.
             return float('inf') # Indicate an invalid state

        current_lift_level = self.floor_levels.get(current_lift_floor)
        if current_lift_level is None:
             # This should not happen if floor_levels was built correctly and state is valid
             # A state with lift at an unknown floor is invalid.
             return float('inf') # Indicate an invalid state

        total_cost = 0

        # Iterate through all passengers that need to be served (those in the goal)
        for passenger in self.goal_passengers:
            # Check if passenger is already served
            if f"(served {passenger})" in state:
                continue # Passenger is served, cost is 0 for this passenger

            # Check if passenger is boarded
            if f"(boarded {passenger})" in state:
                # Passenger is boarded, needs to go to destination and depart
                destin_floor = self.passenger_destinations.get(passenger)
                if destin_floor is None:
                     # This passenger is in the goal but has no destination? Invalid problem.
                     return float('inf') # Indicate an invalid problem setup

                destin_level = self.floor_levels.get(destin_floor)
                if destin_level is None:
                     # Destination floor is unknown? Invalid problem setup.
                     return float('inf') # Indicate an invalid problem setup

                # Cost = move from current lift floor to destination + depart
                movement_cost = abs(current_lift_level - destin_level)
                depart_cost = 1
                total_cost += movement_cost + depart_cost

            else:
                # Passenger is waiting at an origin floor
                # Find the current origin floor from the state
                current_origin_floor = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == "origin" and len(parts) == 3 and parts[1] == passenger:
                        current_origin_floor = parts[2]
                        break

                if current_origin_floor is None:
                    # This passenger is a goal passenger, not served, not boarded, and not at any origin in state?
                    # This indicates an invalid state or problem definition.
                    return float('inf') # Indicate an invalid state

                origin_level = self.floor_levels.get(current_origin_floor)
                if origin_level is None:
                     # Origin floor is unknown? Invalid problem setup.
                     return float('inf') # Indicate an invalid problem setup

                destin_floor = self.passenger_destinations.get(passenger)
                if destin_floor is None:
                     # This passenger is in the goal but has no destination? Invalid problem.
                     return float('inf') # Indicate an invalid problem setup

                destin_level = self.floor_levels.get(destin_floor)
                if destin_level is None:
                     # Destination floor is unknown? Invalid problem setup.
                     return float('inf') # Indicate an invalid problem setup


                # Cost = move from current lift floor to origin + board + move from origin to destination + depart
                move_to_origin_cost = abs(current_lift_level - origin_level)
                board_cost = 1
                move_to_destin_cost = abs(origin_level - destin_level)
                depart_cost = 1

                total_cost += move_to_origin_cost + board_cost + move_to_destin_cost + depart_cost

        return total_cost
