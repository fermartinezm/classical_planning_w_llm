import fnmatch
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or raise an error
        # For robustness, return empty list or handle gracefully
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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class
# Inherit from Heuristic base class if available, otherwise just define the class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not found
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found or implemented.")


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the remaining cost by summing the estimated cost for each unserved passenger independently.
    For an unboarded passenger: 2 actions (board + depart) + movement from current lift floor to origin + movement from origin to destination.
    For a boarded passenger: 1 action (depart) + movement from current lift floor to destination.
    Movement cost between floors is the absolute difference in their floor levels.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor level mapping based on 'above' predicates.
        - Passenger destinations.
        - List of all goal passengers.
        """
        super().__init__(task) # Call base class constructor if inheriting

        self.goals = task.goals
        self.static_facts = task.static

        # Build floor level mapping
        # We need to find the linear order of floors from 'above' facts.
        # Example: (above f2 f1), (above f3 f2) -> f1 < f2 < f3
        floor_above_map = {} # Maps floor_below -> floor_above
        all_mentioned_floors = set()

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above' and len(parts) == 3:
                f_above, f_below = parts[1:]
                floor_above_map[f_below] = f_above
                all_mentioned_floors.add(f_above)
                all_mentioned_floors.add(f_below)
            elif parts and parts[0] == 'destin' and len(parts) == 3:
                 # Also collect floors mentioned in destinations
                 all_mentioned_floors.add(parts[2])

        # Find the lowest floor: it's a floor mentioned, but no other floor is 'above' it.
        # In a linear chain f1 < f2 < ... < fn, fn is above fn-1, ..., f2 is above f1.
        # f1 is the lowest. It is not a *value* in the floor_above_map.
        floors_that_are_above_something = set(floor_above_map.values())
        lowest_floor = None
        for f in all_mentioned_floors:
            if f not in floors_that_are_above_something:
                lowest_floor = f
                break

        self.floor_level = {}
        if lowest_floor:
            current_floor = lowest_floor
            level = 1
            self.floor_level[current_floor] = level
            # Build levels upwards following the 'above' chain
            while current_floor in floor_above_map:
                next_floor = floor_above_map[current_floor]
                level += 1
                self.floor_level[next_floor] = level
                current_floor = next_floor
        # Note: If there are floors not connected by 'above' facts, or multiple chains,
        # this mapping might be incomplete. Assumes a single linear floor structure.
        # If lowest_floor is None but there are mentioned floors, it might be a single-floor problem.
        elif all_mentioned_floors:
             # Assume the single mentioned floor is level 1
             self.floor_level[list(all_mentioned_floors)[0]] = 1


        # Store passenger destinations
        self.passenger_destin = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                passenger, destin_floor = get_parts(fact)[1:]
                self.passenger_destin[passenger] = destin_floor

        # Store all goal passengers
        self.goal_passengers = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                passenger = get_parts(goal)[1]
                self.goal_passengers.add(passenger)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Find current lift floor and its level
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or not mapped, return a high value
        current_level = self.floor_level.get(current_lift_floor)
        if current_level is None:
             # This indicates an issue with floor mapping or state.
             # Return a large value to prune this path.
             return float('inf')


        total_heuristic = 0

        # Iterate through all passengers that need to be served
        for passenger in self.goal_passengers:
            # Check if the passenger is already served
            if f"(served {passenger})" in state:
                continue # This passenger is done

            # Check if the passenger is boarded
            if f"(boarded {passenger})" in state:
                # Passenger is boarded, needs to be dropped off at destination
                destin_floor = self.passenger_destin.get(passenger)
                if destin_floor is None:
                    # Destination unknown for a goal passenger? Invalid problem.
                    return float('inf')

                destin_level = self.floor_level.get(destin_floor)
                if destin_level is None:
                     # Destination floor not mapped? Invalid problem.
                     return float('inf')

                # Cost for a boarded passenger: 1 (depart) + movement from current to destination
                total_heuristic += 1 + abs(current_level - destin_level)

            else:
                # Passenger is unboarded, must be at their origin floor
                # Find the origin floor in the current state
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        break

                if origin_floor is None:
                    # Unserved, not boarded, but no origin fact? Invalid state.
                    return float('inf')

                destin_floor = self.passenger_destin.get(passenger)
                if destin_floor is None:
                    # Destination unknown for a goal passenger? Invalid problem.
                    return float('inf')

                origin_level = self.floor_level.get(origin_floor)
                destin_level = self.floor_level.get(destin_floor)

                if origin_level is None or destin_level is None:
                     # Origin or destination floor not mapped? Invalid problem.
                     return float('inf')

                # Cost for an unboarded passenger:
                # 1 (board) + 1 (depart) + movement from current to origin + movement from origin to destination
                total_heuristic += 2 + abs(current_level - origin_level) + abs(origin_level - destin_level)

        return total_heuristic
