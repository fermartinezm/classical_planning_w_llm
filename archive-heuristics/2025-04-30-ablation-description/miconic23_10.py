from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base is not found,
    # to allow the code structure to be correct for the user's environment.
    # In a real scenario, this import must succeed.
    print("Warning: Could not import heuristics.heuristic_base.Heuristic. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not properly imported or defined.")

# Utility function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input format - maybe return empty list or raise error
        # For PDDL facts represented as strings, this format is expected.
        # Let's assume valid input based on problem description.
        return fact[1:-1].split()
    return fact[1:-1].split()

# Utility function to match facts (similar to Logistics example)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, considering wildcards.
    # Use zip to handle cases where parts might be longer than args (e.g., extra arguments in fact)
    # and fnmatch handles wildcards.
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It sums the estimated cost for each unserved passenger,
    considering their current state (waiting at origin or boarded) and the
    lift's current location. The cost includes boarding/departing actions and
    estimated lift movements based on floor distances.

    # Assumptions
    - All actions (board, depart, up, down) have a unit cost of 1.
    - The floor structure is a simple linear sequence defined by 'above' predicates.
    - The heuristic calculates the minimum number of moves between floors as the
      absolute difference in their assigned indices.
    - The heuristic sums costs for passengers independently, ignoring potential
      synergies (e.g., picking up/dropping off multiple passengers on the same trip).
      This makes it potentially non-admissible but can be effective for greedy search.

    # Heuristic Initialization
    - Parses the 'above' predicates from static facts to determine the linear
      order of floors and assign a numerical index to each floor.
    - Stores the destination floor for each passenger from the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the lift.
    2. For each passenger whose destination is known from the goal:
       a. Check if the passenger is already 'served' (goal condition). If yes, cost is 0 for this passenger.
       b. If not served, find the passenger's destination floor from the stored goal information.
       c. Determine the passenger's current state:
          - Waiting at origin floor: Find the origin floor from the current state's 'origin' facts.
          - Boarded in the lift: The passenger is in the lift, and their 'origin' fact is removed.
       d. Calculate the estimated cost for this passenger:
          - If waiting at origin 'o':
            - Cost = distance(lift_current_floor, o) + 1 (board) + distance(o, destination) + 1 (depart).
          - If boarded:
            - Cost = distance(lift_current_floor, destination) + 1 (depart).
       e. The distance between two floors is the absolute difference of their assigned indices.
    3. Sum the estimated costs for all unserved passengers.
    4. The total sum is the heuristic value. If the sum is 0, it implies all passengers are served, which is the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        super().__init__(task) # Call the base class constructor

        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Determine floor order and assign indices.
        # Build a map: floor_lower -> floor_higher based on (above floor_higher floor_lower)
        floor_below_to_above = {}
        all_floors = set()
        floors_that_are_above_something = set() # Floors that appear as f_higher

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_higher, f_lower = get_parts(fact)[1:]
                floor_below_to_above[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                floors_that_are_above_something.add(f_higher)

        # Find the lowest floor: the one that is in all_floors but not in floors_that_are_above_something.
        potential_lowest_floors = list(all_floors - floors_that_are_above_something)

        if len(potential_lowest_floors) != 1:
             # Handle cases with disconnected floors or invalid structure
             # If there's only one floor total, it's the lowest.
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # This indicates a non-linear or disconnected floor structure,
                 # or a problem where not all floors are linked by 'above'.
                 # The linear distance calculation won't work correctly.
                 # For this problem, assume a valid linear structure.
                 # If multiple potential lowest floors, something is wrong.
                 # Returning inf will make states from such problems have high heuristic.
                 self.floor_indices = {} # Indicate invalid floor setup
                 self.goal_locations = {} # Cannot proceed reliably
                 print(f"Error: Could not determine a single lowest floor from 'above' facts. Found: {potential_lowest_floors}. Heuristic will return inf.")
                 return # Exit init early

        else:
             lowest_floor = potential_lowest_floors[0]


        # Build the ordered list of floors starting from the lowest.
        floor_order = [lowest_floor]
        current_floor = lowest_floor
        while current_floor in floor_below_to_above:
            next_floor = floor_below_to_above[current_floor]
            floor_order.append(next_floor)
            current_floor = next_floor

        # Check if we included all floors.
        if len(floor_order) != len(all_floors):
             # This means some floors were not reachable from the lowest floor
             # following the 'directly above' links. Invalid structure.
             self.floor_indices = {} # Indicate invalid floor setup
             self.goal_locations = {} # Cannot proceed reliably
             print("Error: Floor ordering is incomplete. Not all floors are connected linearly. Heuristic will return inf.")
             return # Exit init early


        self.floor_indices = {floor: i for i, floor in enumerate(floor_order)}

        # Store goal locations for each passenger.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is (served ?p)
            # We need the destination from static facts (destin ?p ?f)
            if match(goal, "served", "*"):
                 passenger_name = get_parts(goal)[1]
                 # Find the corresponding destin fact in static facts
                 destin_fact = next(
                     (fact for fact in static_facts if match(fact, "destin", passenger_name, "*")),
                     None
                 )
                 if destin_fact:
                     self.goal_locations[passenger_name] = get_parts(destin_fact)[2]
                 else:
                     # This shouldn't happen in valid problems, but handle defensively
                     print(f"Warning: Could not find destination for passenger {passenger_name} in static facts.")
                     # Cannot calculate heuristic for this passenger without destination
                     # Skip this passenger.
                     pass # Skip this passenger


    def get_floor_index(self, floor_name):
        """Helper to get the numerical index for a floor name."""
        # Return None if floor_indices is empty (due to init failure) or floor_name not found.
        return self.floor_indices.get(floor_name, None)

    def get_distance(self, floor1_name, floor2_name):
        """Helper to calculate the distance between two floors."""
        # If floor_indices is empty (due to init error), return inf.
        if not self.floor_indices:
             return float('inf')

        idx1 = self.get_floor_index(floor1_name)
        idx2 = self.get_floor_index(floor2_name)

        # If either floor name wasn't found in the indices map (shouldn't happen
        # if all floors were processed correctly in init), return inf.
        if idx1 is None or idx2 is None:
            print(f"Warning: Could not find index for floor {floor1_name} or {floor2_name}. Returning inf distance.")
            return float('inf')

        return abs(idx1 - idx2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If init failed to build floor indices, return inf for all states.
        if not self.floor_indices:
             return float('inf')

        # Check if the goal is reached.
        if self.goals <= state:
             return 0

        total_cost = 0  # Initialize action cost counter.

        # Find the lift's current floor.
        lift_current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_current_floor = get_parts(fact)[1]
                break
        if lift_current_floor is None:
             # Should not happen in a valid state, but handle defensively
             print("Warning: Lift location not found in state. Returning inf.")
             return float('inf')


        # Iterate through all passengers whose destinations are known from goals.
        # Only consider passengers whose destinations were successfully extracted in init.
        for passenger_name, goal_location in self.goal_locations.items():
            # Check if the passenger is already served.
            if f"(served {passenger_name})" in state:
                continue # This passenger is done, cost is 0 for them.

            # Passenger is not served. Find their current state.
            is_boarded = f"(boarded {passenger_name})" in state
            origin_location = None
            if not is_boarded:
                # Find the origin floor if not boarded.
                # An unserved, unboarded passenger must have an origin fact.
                origin_fact = next(
                    (fact for fact in state if match(fact, "origin", passenger_name, "*")),
                    None
                )
                if origin_fact:
                    origin_location = get_parts(origin_fact)[2]
                else:
                     # This indicates an invalid state representation or problem definition.
                     # A passenger who is not served and not boarded must have an origin.
                     # Return inf to indicate this path is likely invalid.
                     print(f"Warning: Passenger {passenger_name} not served, not boarded, and no origin fact found. Returning inf.")
                     return float('inf')


            # Calculate cost for this unserved passenger.
            if not is_boarded:
                # Passenger is waiting at origin_location.
                # Needs: move lift to origin, board, move lift to destination, depart.
                cost_for_passenger = 0
                # 1. Move lift to origin
                dist_to_origin = self.get_distance(lift_current_floor, origin_location)
                if dist_to_origin == float('inf'): return float('inf') # Propagate error
                cost_for_passenger += dist_to_origin

                # 2. Board action
                cost_for_passenger += 1

                # 3. Move lift from origin to destination
                dist_origin_to_dest = self.get_distance(origin_location, goal_location)
                if dist_origin_to_dest == float('inf'): return float('inf') # Propagate error
                cost_for_passenger += dist_origin_to_dest

                # 4. Depart action
                cost_for_passenger += 1

                total_cost += cost_for_passenger

            else: # Passenger is boarded.
                # Needs: move lift to destination, depart.
                cost_for_passenger = 0
                # 1. Move lift to destination
                dist_to_dest = self.get_distance(lift_current_floor, goal_location)
                if dist_to_dest == float('inf'): return float('inf') # Propagate error
                cost_for_passenger += dist_to_dest

                # 2. Depart action
                cost_for_passenger += 1

                total_cost += cost_for_passenger

        # The total_cost is the sum of estimated costs for all unserved passengers.
        # If total_cost is 0, it means all passengers are served, which is the goal.
        # If total_cost is > 0, it's a non-goal state.
        # This satisfies the requirement that heuristic is 0 only for goal states.

        return total_cost
