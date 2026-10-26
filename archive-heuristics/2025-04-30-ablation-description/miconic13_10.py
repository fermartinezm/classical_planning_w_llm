import re
from fnmatch import fnmatch
# Assuming heuristic_base.py is available in a 'heuristics' directory
# and defines a base class with __init__ and __call__ methods.
# If running standalone, you might need a dummy base class:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format gracefully
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts
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
    This heuristic estimates the cost to serve all passengers. It sums the estimated
    cost for each passenger who has not yet been served. The cost for a passenger
    depends on whether they are waiting at their origin floor or are already boarded.
    The cost includes travel time for the lift (estimated by floor difference) and
    the cost of board/depart actions.

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < f3).
      The sorting assumes floor names end with a number that determines their level.
    - The cost of moving the lift one floor is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic calculates the cost for each passenger independently, ignoring
      potential optimizations like picking up/dropping off multiple passengers
      at the same floor in one trip. This makes it a relaxation.

    # Heuristic Initialization
    - Extracts the ordered list of floors and creates a mapping from floor name to index.
      This is done by collecting all floor objects and sorting them based on the
      numerical suffix found in their names.
    - Extracts the origin and destination floors for each passenger from the initial state.
    - Identifies the set of passengers that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state of the world.
    2. Find the current floor of the lift.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each passenger that is required to be served in the goal state.
    5. For the current passenger:
       a. Check if the passenger is already served in the current state. If yes, add 0 cost for this passenger and continue to the next.
       b. If the passenger is not served, find their original origin floor and final destination floor (stored during initialization).
       c. Check if the passenger is currently boarded on the lift.
       d. If the passenger is boarded:
          - Calculate the floor distance between the current lift location and the passenger's destination floor using the floor index map.
          - Add this distance to the total cost.
          - Add 1 to the total cost for the 'depart' action.
       e. If the passenger is not boarded (meaning they are still at their origin floor):
          - Calculate the floor distance between the current lift location and the passenger's origin floor using the floor index map.
          - Add this distance to the total cost.
          - Add 1 to the total cost for the 'board' action.
          - Calculate the floor distance between the passenger's origin floor and their destination floor using the floor index map.
          - Add this distance to the total cost.
          - Add 1 to the total cost for the 'depart' action.
    6. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger origins/destinations,
        and goal passengers from the task definition.
        """
        self.goals = task.goals  # Goal conditions
        self.initial_state = task.initial_state # Initial state facts
        self.static_facts = task.static # Static facts

        # 1. Extract floor ordering and create floor index map
        # Collect all unique floor names from static facts (above) and initial state (lift-at, origin, destin).
        floor_names = set()
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'above' and len(parts) == 3:
                 floor_names.add(parts[1])
                 floor_names.add(parts[2])
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ['lift-at', 'origin', 'destin'] and len(parts) >= 2:
                 # Floor is typically the last argument for these predicates
                 floor_names.add(parts[-1])

        # Sort floor names numerically based on the number part (e.g., f1, f10, f2 -> f1, f2, f10)
        # Use regex to find the number at the end of the string.
        def sort_key(floor_name):
            match = re.search(r'\d+$', floor_name)
            if match:
                return int(match.group(0))
            # If no number found, assign a large value to put it at the end
            # This handles cases like 'ground', 'first', etc. if they occurred.
            return float('inf')

        sorted_floor_names = sorted(list(floor_names), key=sort_key)
        self.floor_indices = {floor_name: i for i, floor_name in enumerate(sorted_floor_names)}

        # 2. Extract origin and destination floors for each passenger from the initial state
        self.origin_floors = {}
        self.destin_floors = {}
        self.all_passengers = set() # Keep track of all passengers mentioned in init

        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'origin' and len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    self.origin_floors[passenger] = floor
                    self.all_passengers.add(passenger)
                elif parts[0] == 'destin' and len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    self.destin_floors[passenger] = floor
                    self.all_passengers.add(passenger)

        # 3. Identify goal passengers (those who need to be served)
        self.goal_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served' and len(parts) == 2:
                self.goal_passengers.add(parts[1])

        # Only consider passengers that are in the goals AND have origin/destin in initial state
        # This filters out any passengers mentioned only in static facts or not relevant to the goal.
        self.goal_passengers = self.goal_passengers.intersection(self.all_passengers)


    def distance(self, floor1, floor2):
        """Calculate the floor distance between two floors."""
        if floor1 not in self.floor_indices or floor2 not in self.floor_indices:
             # This indicates an issue with floor parsing or an invalid state/fact.
             # Returning infinity is a safe way to indicate an unreachable or problematic state.
             return float('inf')
        return abs(self.floor_indices[floor1] - self.floor_indices[floor2])

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all passengers.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # Find the current lift location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # The lift location is unknown in this state. This state is likely invalid or unreachable.
             return float('inf')

        total_cost = 0  # Initialize action cost counter.

        # Iterate through passengers who need to be served according to the goal
        for passenger in self.goal_passengers:
            # Check if the passenger is already served in the current state
            if f'(served {passenger})' in state:
                continue # This passenger is done

            # Get origin and destination floors (stored in __init__)
            # We assume passengers in goal have origin/destin in initial state.
            # The check in __init__ ensures we only process such passengers.
            # If for some reason a goal passenger doesn't have origin/destin (malformed problem),
            # the intersection in __init__ would exclude them, which is appropriate.
            origin_floor = self.origin_floors.get(passenger)
            destin_floor = self.destin_floors.get(passenger)

            # This check should ideally not be needed if __init__ filters correctly,
            # but added for robustness against unexpected states/goals.
            if origin_floor is None or destin_floor is None:
                 # Goal passenger has no origin or destination defined in init.
                 # This is likely a malformed problem or an unreachable state.
                 return float('inf')


            # Check if the passenger is currently boarded
            if f'(boarded {passenger})' in state:
                # Passenger is on the lift, needs to go to destination and depart
                # Cost = travel distance + depart action
                cost_for_passenger = self.distance(current_lift_floor, destin_floor) + 1
                # If distance is inf, total_cost becomes inf
                if cost_for_passenger == float('inf'): return float('inf')
                total_cost += cost_for_passenger
            else:
                # Passenger is waiting at origin (assuming valid state).
                # Needs lift to come to origin, board, travel to destin, and depart.
                # Cost = travel to origin + board + travel to destin + depart
                cost_to_origin = self.distance(current_lift_floor, origin_floor)
                cost_origin_to_destin = self.distance(origin_floor, destin_floor)

                if cost_to_origin == float('inf') or cost_origin_to_destin == float('inf'):
                     return float('inf') # Cannot calculate distance

                cost_for_passenger = cost_to_origin + 1 + cost_origin_to_destin + 1
                total_cost += cost_for_passenger

        return total_cost
