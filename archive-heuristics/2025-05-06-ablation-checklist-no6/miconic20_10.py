from fnmatch import fnmatch
# Assuming Heuristic base class is provided in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not automatically available,
# you might need a placeholder like this:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args if no wildcards are used
    # or handle cases where pattern is shorter than fact parts (e.g., matching predicate only)
    # A simpler approach for this domain is to just zip and check.
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unserved passenger independently.

    For a waiting passenger P at origin F_o with destination F_d:
    Estimated cost = travel(current_lift_floor, F_o) + 1 (board) + travel(F_o, F_d) + 1 (depart)

    For a boarded passenger P with destination F_d:
    Estimated cost = travel(current_lift_floor, F_d) + 1 (depart)

    Travel cost between floors is the absolute difference in floor indices.
    This heuristic is non-admissible as it sums independent costs, ignoring
    shared lift travel and capacity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor ordering to calculate distances.
        - Passenger destinations.
        - Goal passengers.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract floor objects and create floor mapping based on 'above' facts.
        # Assumes floor names are like f<number> and (above fX fY) implies X < Y.
        floor_names = set()
        for fact in self.static_facts:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                floor_names.add(parts[1])
                floor_names.add(parts[2])

        # Sort floors based on the number part of their name (e.g., f1, f2, f10)
        # Handle potential errors if floor names don't follow f<number> format.
        try:
            sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
        except ValueError:
             # Fallback or error handling if floor names are not f<number>
             # For this problem, we assume f<number> format is standard.
             # A more robust approach would build a graph from 'above' facts.
             print("Warning: Floor names might not be in 'f<number>' format. Sorting might be incorrect.")
             sorted_floor_names = sorted(list(floor_names)) # Simple alphabetical sort as fallback

        self.floor_to_index = {floor_name: i for i, floor_name in enumerate(sorted_floor_names)}
        self.index_to_floor = {i: floor_name for floor_name, i in self.floor_to_index.items()}

        # Extract passenger destinations (static)
        self.passenger_destinations = {}
        # Identify all goal passengers first
        self.goal_passengers = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "served"}

        # Find destinations for all goal passengers from static facts
        for fact in self.static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, destin_floor = get_parts(fact)
                 if passenger in self.goal_passengers:
                     self.passenger_destinations[passenger] = destin_floor

    def get_floor_index(self, floor_name):
        """Returns the numerical index for a given floor name."""
        # Return a default value like -1 or raise an error if floor_name is not in the map.
        # Assuming all relevant floors are in the static facts used for mapping.
        return self.floor_to_index.get(floor_name) # None if not found

    def get_distance(self, floor1_name, floor2_name):
        """Calculates the distance (number of moves) between two floors."""
        index1 = self.get_floor_index(floor1_name)
        index2 = self.get_floor_index(floor2_name)
        if index1 is None or index2 is None:
            # This indicates an issue if a floor from state/goals is not in static 'above' facts.
            # Return a large value to penalize states with unknown floors, or 0 if assuming valid states.
            # Assuming valid states where all floors are known.
            # print(f"Warning: Unknown floor encountered: {floor1_name} or {floor2_name}")
            return 0 # Or float('inf') depending on desired behavior for invalid states
        return abs(index1 - index2)

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.

        The heuristic is the sum of estimated costs for each unserved
        goal passenger.
        """
        state = node.state

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # Assuming 'lift-at' predicate is always present in a valid state.
        if current_lift_floor is None:
             # Should not happen in a valid state
             return float('inf') # Or handle appropriately

        total_heuristic = 0

        # Iterate over all goal passengers
        for passenger in self.goal_passengers:
            # Check if the passenger is already served
            if f"(served {passenger})" in state:
                continue # This passenger is done

            # Get the passenger's destination floor
            dest_floor = self.passenger_destinations.get(passenger)
            if dest_floor is None:
                 # This passenger is a goal passenger but has no destination in static facts?
                 # Should not happen in a well-formed problem. Ignore or penalize.
                 # print(f"Warning: Goal passenger {passenger} has no destination defined.")
                 continue

            # Check if the passenger is currently boarded
            if f"(boarded {passenger})" in state:
                # Passenger is boarded, needs to travel to destination and depart
                # Cost = travel(current_lift_floor, dest_floor) + 1 (depart)
                h_p = self.get_distance(current_lift_floor, dest_floor) + 1
                total_heuristic += h_p
            else:
                # Passenger is not served and not boarded, must be waiting at origin
                # Find the passenger's origin floor in the current state
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        break

                if origin_floor is None:
                    # Passenger is unserved, not boarded, and not waiting?
                    # This implies an invalid state representation or problem definition.
                    # For a valid state derived from initial state, an unserved, non-boarded
                    # goal passenger must have an (origin p f) fact.
                    # print(f"Warning: Unserved, non-boarded goal passenger {passenger} has no origin fact in state.")
                    # Assume they are unreachable or add a large penalty. Returning 0 might hide issues.
                    # Let's assume valid states and this branch is not reached for solvable problems.
                    pass # Skip this passenger or add penalty if state is unexpected
                else:
                    # Passenger is waiting at origin, needs pickup and dropoff
                    # Cost = travel(current_lift_floor, origin_floor) + 1 (board) + travel(origin_floor, dest_floor) + 1 (depart)
                    h_p = self.get_distance(current_lift_floor, origin_floor) + 1 + self.get_distance(origin_floor, dest_floor) + 1
                    total_heuristic += h_p

        return total_heuristic

