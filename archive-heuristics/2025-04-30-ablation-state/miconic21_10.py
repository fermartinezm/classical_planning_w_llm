# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Define the heuristic class
# If a base class 'Heuristic' is provided by the planner framework,
# inherit from it like: class miconicHeuristic(Heuristic):
class miconicHeuristic:
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers by summing the estimated actions needed for each unserved
    passenger independently. It considers the lift's current position,
    the passenger's current state (at origin or boarded), and their
    origin and destination floors.

    # Assumptions
    - All actions (board, depart, up, down) have a cost of 1.
    - The distance between floors is the absolute difference in their levels.
    - The heuristic calculates the cost for each passenger as if they were
      transported individually, ignoring the possibility of carrying multiple
      passengers simultaneously. This makes the heuristic non-admissible but
      potentially more informative for greedy search.

    # Heuristic Initialization
    - Extracts all floor objects and determines their numerical order (level)
      based on their names (assuming 'f1', 'f2', etc.). A mapping from floor
      name to its level is created.
    - Extracts the destination floor for each passenger from the static
      'destin' facts. A mapping from passenger name to destination floor is created.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Identify which passengers are currently served, boarded, or waiting at their origin floor.
    3. Initialize the total heuristic cost to 0.
    4. For each passenger that needs to be served (i.e., has a destination defined in the problem):
        a. If the passenger is already served, they contribute 0 to the heuristic.
        b. If the passenger is currently boarded in the lift:
            - Calculate the floor distance between the lift's current floor and the passenger's destination floor.
            - Add this distance plus 1 (for the 'depart' action) to the total cost.
        c. If the passenger is currently waiting at their origin floor:
            - Find the passenger's origin floor.
            - Calculate the floor distance between the lift's current floor and the passenger's origin floor.
            - Add this distance plus 1 (for the 'board' action).
            - Calculate the floor distance between the passenger's origin floor and their destination floor.
            - Add this distance plus 1 (for the 'depart' action).
            - Sum these costs and add them to the total cost.
        d. A passenger not served, not boarded, and not at their origin is in an invalid state according to the domain.
    5. The total heuristic value is the sum of costs calculated for all unserved passengers.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.

        Args:
            task: The planning task object containing initial state, goals, and static facts.
        """
        # Extract all floor names and determine their levels
        all_facts = set(task.static) | set(task.initial_state)

        floor_names = set()
        for fact in all_facts:
            parts = get_parts(fact)
            # Check predicates that involve floors
            if parts and parts[0] in ["lift-at", "origin", "destin"]:
                # Floor is the last argument
                if len(parts) > 1: # Ensure there's an argument
                    floor_names.add(parts[-1])
            elif parts and parts[0] == "above":
                # Floors are the two arguments
                if len(parts) > 2: # Ensure there are two arguments
                    floor_names.add(parts[1])
                    floor_names.add(parts[2])

        # Sort floor names numerically (assuming names are like 'f<number>')
        # Handle potential errors if floor names are not in expected format
        try:
            # Filter out any names that don't start with 'f' followed by digits
            valid_floor_names = [f for f in floor_names if f.startswith('f') and f[1:].isdigit()]
            sorted_floor_names = sorted(valid_floor_names, key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # If sorting fails, the map will be empty or incorrect, leading to errors later.
             # Re-raising the error is appropriate if the format is strictly expected.
             raise ValueError("Floor names are not in the expected 'f<number>' format.")


        # Create floor_to_level map (using 1-based indexing)
        self.floor_to_level = {floor: i + 1 for i, floor in enumerate(sorted_floor_names)}

        # Extract passenger_to_destin map from static facts
        self.passenger_to_destin = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.passenger_to_destin[passenger] = floor


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining cost.
        """
        state = node.state

        # Extract relevant information from the current state
        current_lift_floor = None
        served_passengers = set()
        boarded_passengers = set()
        origin_locations = {} # map passenger to origin floor

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            if predicate == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
            elif predicate == "served" and len(parts) == 2:
                served_passengers.add(parts[1])
            elif predicate == "boarded" and len(parts) == 2:
                boarded_passengers.add(parts[1])
            elif predicate == "origin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                origin_locations[passenger] = floor

        # If for some reason the lift location is unknown, the heuristic is infinite (or a large number)
        # This shouldn't happen in valid states for this domain.
        if current_lift_floor is None:
             # This indicates an invalid state for this domain.
             # Return infinity to guide search away.
             return float('inf')


        total_cost = 0

        # Iterate through all passengers that need to reach a destination
        for passenger, f_dest in self.passenger_to_destin.items():
            # If the passenger is already served, they require no more actions
            if passenger in served_passengers:
                continue

            # Calculate cost based on passenger's current state
            if passenger in boarded_passengers:
                # Passenger is boarded, needs to go to destination and depart
                f_current_passenger_loc = current_lift_floor # Passenger is with the lift
                # Ensure floors are in the level map (should be if parsing was correct)
                if f_current_passenger_loc in self.floor_to_level and f_dest in self.floor_to_level:
                    cost_move_to_dest = abs(self.floor_to_level[f_current_passenger_loc] - self.floor_to_level[f_dest])
                    cost_depart = 1
                    total_cost += cost_move_to_dest + cost_depart
                else:
                    # Should not happen in valid problems/states if floor parsing is correct
                    # Handle as potentially unsolvable or high cost
                    return float('inf') # Indicate unsolvable or error state

            elif passenger in origin_locations:
                # Passenger is at origin, needs lift to come, board, go to dest, depart
                f_origin = origin_locations[passenger]
                 # Ensure floors are in the level map
                if current_lift_floor in self.floor_to_level and f_origin in self.floor_to_level and f_dest in self.floor_to_level:
                    cost_move_to_origin = abs(self.floor_to_level[current_lift_floor] - self.floor_to_level[f_origin])
                    cost_board = 1
                    cost_move_origin_to_dest = abs(self.floor_to_level[f_origin] - self.floor_to_level[f_dest])
                    cost_depart = 1
                    total_cost += cost_move_to_origin + cost_board + cost_move_origin_to_dest + cost_depart
                else:
                     # Should not happen
                     return float('inf') # Indicate unsolvable or error state

            # Note: A passenger not served, not boarded, and not at origin
            # is not possible in a valid state of this domain.

        return total_cost
