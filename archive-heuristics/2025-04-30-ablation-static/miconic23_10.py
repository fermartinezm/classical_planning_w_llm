# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
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
    This heuristic estimates the number of actions required to serve all passengers.
    For each unserved passenger, it calculates the cost to:
    1. Move the lift to the passenger's origin floor (if not already boarded).
    2. Board the passenger (if not already boarded).
    3. Move the lift to the passenger's destination floor.
    4. Depart the passenger at the destination.
    The total heuristic is the sum of these costs for all unserved passengers.

    # Assumptions
    - The lift has infinite capacity.
    - The order in which passengers are picked up or dropped off does not affect the cost calculation for other passengers (this is a simplification).
    - The 'above' predicate defines the immediate adjacency of floors, and forms a linear sequence of floors.
    - The actions 'up' and 'down' move the lift between immediately adjacent floors as defined by the 'above' predicate, following the PDDL definition (even if counter-intuitive names).
    - Passenger destinations are static and available in the initial state.

    # Heuristic Initialization
    - Extracts the floor structure from the 'above' facts in the initial state to create a mapping from floor names to numerical indices. This allows calculating the distance between floors.
    - Extracts the destination floor for each passenger from the initial state.
    - Identifies all passengers involved in the problem based on initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. For each passenger:
       a. Check if the passenger is already 'served'. If yes, the cost for this passenger is 0.
       b. If not 'served', check if the passenger is 'boarded'.
       c. If 'boarded':
          - Get the passenger's destination floor.
          - Calculate the distance between the current lift floor and the destination floor (number of move actions).
          - Add 1 for the 'depart' action. This is the cost for this passenger.
       d. If not 'boarded' (implies the passenger is at their origin floor, i.e., '(origin p f_origin)' is true):
          - Find the passenger's origin floor (from the state).
          - Get the passenger's destination floor.
          - Calculate the distance between the current lift floor and the origin floor.
          - Add 1 for the 'board' action.
          - Calculate the distance between the origin floor and the destination floor.
          - Add 1 for the 'depart' action.
          - Sum these values. This is the cost for this passenger.
    3. The total heuristic value is the sum of the costs calculated for all passengers.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor structure and passenger destinations.
        """
        super().__init__(task) # Call parent constructor if needed
        self.task = task
        self.goals = task.goals # Goal conditions (served passengers)

        # Extract floor structure and build floor_to_index map
        # Assuming 'above' facts are in initial_state and define adjacency
        above_facts = [fact for fact in task.initial_state if match(fact, "above", "*", "*")]

        below_map = {}
        above_floors_set = set()
        below_floors_set = set()

        for fact in above_facts:
            parts = get_parts(fact)
            if len(parts) == 3: # Ensure fact has correct structure (above f1 f2)
                _, f_above, f_below = parts
                below_map[f_above] = f_below
                above_floors_set.add(f_above)
                below_floors_set.add(f_below)
            # else: ignore malformed above fact

        # Find the top floor (the one that is above another but not below any)
        # Handle case with no above facts (single floor problem)
        top_floor = None
        if not above_floors_set and not below_floors_set:
             # No 'above' facts, likely a single floor problem or malformed
             # Try to find any floor mentioned
             all_floors = set()
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if match(fact, "lift-at", "*") and len(parts) == 2:
                     all_floors.add(parts[1])
                 elif match(fact, "origin", "*", "*") and len(parts) == 3:
                     all_floors.add(parts[2])
                 elif match(fact, "destin", "*", "*") and len(parts) == 3:
                     all_floors.add(parts[2])
             if len(all_floors) == 1:
                 top_floor = list(all_floors)[0]
             # If multiple floors but no 'above', we can't build a map -> inf distance
        else:
            # In a valid linear structure, there should be exactly one top floor
            potential_top_floors = above_floors_set - below_floors_set
            if len(potential_top_floors) == 1:
                top_floor = potential_top_floors.pop()
            # If len != 1, structure is not a simple linear chain -> inf distance

        self.floor_to_index = {}
        if top_floor:
            current_floor = top_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                next_floor = below_map.get(current_floor)
                current_floor = next_floor
                index += 1
        # If top_floor is None, floor_to_index remains empty. get_distance will return inf.

        # Extract passenger destinations and identify all passengers
        self.passenger_destinations = {}
        self.all_passengers = set()

        # Destinations are typically in the initial state
        for fact in task.initial_state:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, passenger, destination = parts
                    self.passenger_destinations[passenger] = destination
                    self.all_passengers.add(passenger)
            elif match(fact, "origin", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                    _, passenger, origin = parts
                    self.all_passengers.add(passenger)

        # Also add passengers from goals if any were missed (e.g., already boarded in init)
        for goal in task.goals:
             if match(goal, "served", "*"):
                 parts = get_parts(goal)
                 if len(parts) == 2:
                    _, passenger = parts
                    self.all_passengers.add(passenger)


    def get_floor_index(self, floor_name):
        """Helper to get the numerical index for a floor name."""
        # Return a large value for unknown floors to make paths through them expensive
        return self.floor_to_index.get(floor_name, float('inf'))

    def get_distance(self, floor1, floor2):
        """Calculate the number of moves between two floors."""
        index1 = self.get_floor_index(floor1)
        index2 = self.get_floor_index(floor2)
        if index1 == float('inf') or index2 == float('inf'):
             return float('inf') # Cannot calculate distance if floors are unknown
        return abs(index1 - index2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Check if goal is reached
        # This check is redundant if the search algorithm checks goal states,
        # but good practice for a heuristic that should be 0 at goal.
        if self.task.goal_reached(state):
            return 0

        total_heuristic_cost = 0

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, current_lift_floor = parts
                    break
        # If lift location isn't found, something is wrong with the state representation
        if current_lift_floor is None:
             # Cannot proceed without lift location. Indicate high cost.
             return float('inf')

        # Track status of each passenger
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*") and len(get_parts(fact))==2}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*") and len(get_parts(fact))==2}
        origin_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*") and len(get_parts(fact))==3}

        for passenger in self.all_passengers:
            if passenger in served_passengers:
                # Passenger is served, no cost
                continue

            # Passenger is not served
            destin_floor = self.passenger_destinations.get(passenger)
            if destin_floor is None:
                 # Passenger has no destination defined? Should not happen in valid problem.
                 # Treat as impossible to serve, high cost.
                 return float('inf')

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to go to destination and depart
                distance_to_destin = self.get_distance(current_lift_floor, destin_floor)
                if distance_to_destin == float('inf'): return float('inf') # Cannot reach destination
                cost_for_passenger = distance_to_destin + 1 # +1 for depart action
                total_heuristic_cost += cost_for_passenger
            elif passenger in origin_locations:
                # Passenger is waiting at origin, needs pickup and transport
                origin_floor = origin_locations[passenger]
                distance_to_origin = self.get_distance(current_lift_floor, origin_floor)
                distance_origin_to_destin = self.get_distance(origin_floor, destin_floor)
                if distance_to_origin == float('inf') or distance_origin_to_destin == float('inf'):
                     return float('inf') # Cannot reach origin or destination

                cost_for_passenger = (
                    distance_to_origin + # Travel to origin
                    1 + # Board action
                    distance_origin_to_destin + # Travel to destination
                    1 # Depart action
                )
                total_heuristic_cost += cost_for_passenger
            # Else: Passenger is not served, not boarded, and not at origin.
            # This state shouldn't occur in a valid plan trace from initial state.
            # If it occurs, it might indicate an unreachable state or an issue.
            # We assume such states won't be reached or don't contribute to heuristic
            # in a way that helps the search if they are invalid.
            # If we encounter such a passenger, it might indicate an unsolvable state
            # or a state outside the expected problem structure. Returning inf is an option.
            # For now, we assume valid states only have passengers in served, boarded, or origin predicates.

        return total_heuristic_cost
