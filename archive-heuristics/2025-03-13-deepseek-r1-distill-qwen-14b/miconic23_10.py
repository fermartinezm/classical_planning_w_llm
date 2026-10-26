from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements of the lift and the boarding/departing actions.

    # Assumptions:
    - The lift can move one floor at a time.
    - Each boarding and departing action counts as one step.
    - Passengers are served when the lift moves them from their origin to destination and they are departed.

    # Heuristic Initialization
    - Extracts the floor hierarchy from static facts to determine floor levels.
    - Maps each floor to its level for distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current position of the lift.
    2. For each passenger, determine if they are served, boarded, their origin, and destination.
    3. Group passengers into those who are boarded and those who are not.
    4. For each group of unboarded passengers, calculate the cost to move the lift to their origin, board them, move to their destination, and depart them.
    5. For each group of boarded passengers, calculate the cost to move the lift to their destination and depart them.
    6. Sum all these costs to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting floor hierarchy from static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build floor hierarchy
        self.floor_level = {}
        above_graph = {}
        below_graph = {}
        floors = set()

        for fact in static_facts:
            if fact.startswith('(above '):
                parts = get_parts(fact)
                if len(parts) == 2:
                    floor1, floor2 = parts
                    floors.add(floor1)
                    floors.add(floor2)
                    above_graph[floor1] = floor2
                    below_graph[floor2] = floor1

        # Find the lowest floor (not in below_graph)
        lowest_floor = None
        for floor in floors:
            if floor not in below_graph:
                lowest_floor = floor
                break

        # Assign levels starting from the lowest floor
        current_floor = lowest_floor
        level = 1
        self.floor_level[current_floor] = level

        while current_floor in above_graph:
            current_floor = above_graph[current_floor]
            level += 1
            self.floor_level[current_floor] = level

    def __call__(self, node):
        """Compute the estimated number of actions to serve all passengers."""
        state = node.state

        # Extract lift's current position
        lift_position = None
        for fact in state:
            if fact.startswith('(lift-at '):
                parts = get_parts(fact)
                if len(parts) == 1:
                    lift_position = parts[0]
                break

        # Data structures to track passenger information
        served_passengers = set()
        boarded_passengers = set()
        origin_dict = {}
        destination_dict = {}

        for fact in state:
            if fact.startswith('(served '):
                parts = get_parts(fact)
                if len(parts) == 1:
                    p = parts[0]
                    served_passengers.add(p)
            elif fact.startswith('(boarded '):
                parts = get_parts(fact)
                if len(parts) == 1:
                    p = parts[0]
                    boarded_passengers.add(p)
            elif fact.startswith('(origin '):
                parts = get_parts(fact)
                if len(parts) == 2:
                    p, origin = parts
                    origin_dict[p] = origin
            elif fact.startswith('(destin '):
                parts = get_parts(fact)
                if len(parts) == 2:
                    p, destin = parts
                    destination_dict[p] = destin

        # Group passengers
        unboarded_groups = {}
        boarded_groups = {}

        for p in origin_dict:
            if p not in served_passengers:
                if p in boarded_passengers:
                    origin = origin_dict[p]
                    destin = destination_dict[p]
                    key = (origin, destin)
                    boarded_groups[key] = boarded_groups.get(key, 0) + 1
                else:
                    origin = origin_dict[p]
                    destin = destination_dict[p]
                    key = (origin, destin)
                    unboarded_groups[key] = unboarded_groups.get(key, 0) + 1

        total_cost = 0

        # Process unboarded groups
        for (origin, destin), count in unboarded_groups.items():
            # Calculate distance from lift to origin
            if origin not in self.floor_level or lift_position not in self.floor_level:
                distance_lift_to_origin = 0
            else:
                distance_lift_to_origin = abs(self.floor_level[lift_position] - self.floor_level[origin])
            # Calculate distance from origin to destination
            distance_origin_to_destin = abs(self.floor_level[origin] - self.floor_level[destin])
            # Add the cost for this group
            total_cost += distance_lift_to_origin + distance_origin_to_destin + 2 * count

        # Process boarded groups
        for (origin, destin), count in boarded_groups.items():
            # Calculate distance from origin to destination
            distance_origin_to_destin = abs(self.floor_level[origin] - self.floor_level[destin])
            # Add the cost for this group
            total_cost += distance_origin_to_destin + count

        return total_cost
