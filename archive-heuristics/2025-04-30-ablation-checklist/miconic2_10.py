from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and ensure correct splitting
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It calculates the cost for each unserved passenger independently
    and sums these costs. The cost for a single passenger includes:
    1. Moving the lift from its current location to the passenger's origin floor.
    2. Boarding the passenger (1 action).
    3. Moving the lift from the origin floor to the passenger's destination floor.
    4. Departing the passenger (1 action).
    If a passenger is already boarded, the cost is simplified to:
    1. Moving the lift from its current location to the passenger's destination floor.
    2. Departing the passenger (1 action).

    This heuristic is not admissible as it overestimates lift movement costs
    by calculating them independently for each passenger, but it aims to guide
    a greedy search effectively by prioritizing states where passengers are
    closer to being served.

    # Heuristic Initialization
    - Parses static facts to determine passenger destinations.
    - Parses static facts defining the 'above' relationship to build a mapping
      of floors to numerical levels, allowing calculation of distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers and their destinations from static facts.
    2. Determine the numerical level for each floor by parsing the 'above'
       predicates, establishing the floor order.
    3. In the `__call__` method, find the current floor of the lift.
    4. Initialize total heuristic cost to 0.
    5. Iterate through each passenger:
       - Check if the passenger is already 'served' in the current state. If yes, skip.
       - If not served, determine if the passenger is waiting at an origin floor
         or is already 'boarded'.
       - If waiting at an origin floor:
         - Get the origin and destination floors.
         - Calculate the distance the lift needs to travel from its current floor
           to the origin floor (absolute difference in floor levels).
         - Add this distance to the passenger's cost.
         - Add 1 for the 'board' action.
         - Calculate the distance the lift needs to travel from the origin floor
           to the destination floor.
         - Add this distance to the passenger's cost.
         - Add 1 for the 'depart' action.
       - If 'boarded':
         - Get the destination floor.
         - Calculate the distance the lift needs to travel from its current floor
           to the destination floor.
         - Add this distance to the passenger's cost.
         - Add 1 for the 'depart' action.
       - Add the calculated passenger cost to the total heuristic cost.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and
        building the floor level mapping.
        """
        self.goals = task.goals # Store goals for potential future use (though not strictly needed for this heuristic logic)
        static_facts = task.static

        # 1. Extract passenger destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # 2. Build floor level mapping from 'above' predicates
        self.floor_levels = {}
        floor_above_map = {} # Maps a floor to the floor immediately above it
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, floor_high, floor_low = get_parts(fact)
                floor_above_map[floor_low] = floor_high
                all_floors.add(floor_high)
                all_floors.add(floor_low)

        if not all_floors:
             # Handle case with no floors or no above facts (e.g., empty problem)
             return

        # Find the lowest floor (a floor that is not a key in floor_above_map)
        # This assumes 'above' defines a chain from lowest to highest.
        lowest_floor = None
        # Find all floors that are values (i.e., have a floor below them)
        floors_with_floor_below = set(floor_above_map.keys())
        # The lowest floor is one that is in all_floors but not in the keys
        potential_lowest = all_floors - floors_with_floor_below
        if len(potential_lowest) == 1:
             lowest_floor = potential_lowest.pop()
        elif len(potential_lowest) > 1:
             # This indicates a disconnected floor structure or error in PDDL
             # For robustness, find the one that is not a value either?
             # Or just pick one and hope for the best? Let's assume a single chain.
             # A more robust way: find a floor that is never the *second* argument of 'above'
             floors_as_lower = set(f_low for f_high, f_low in floor_above_map.items())
             potential_lowest = all_floors - floors_as_lower
             if len(potential_lowest) == 1:
                 lowest_floor = potential_lowest.pop()
             else:
                 # Fallback: handle simple cases or raise error
                 # If only one floor, it's the lowest
                 if len(all_floors) == 1:
                     lowest_floor = next(iter(all_floors))
                 else:
                     # This heuristic might not work correctly for complex floor structures
                     print("Warning: Could not uniquely determine lowest floor. Heuristic might be inaccurate.")
                     # Attempt to find a floor that is a value but not a key?
                     # This is getting complicated. Let's stick to the standard chain assumption.
                     # Revert to finding floor not in keys, assuming it's the lowest in a chain
                     potential_lowest = all_floors - floors_with_floor_below
                     if potential_lowest:
                         lowest_floor = potential_lowest.pop()
                     else:
                         # Still no lowest floor found, possibly a cycle or empty
                         print("Error: Could not determine lowest floor from 'above' predicates.")
                         self.floor_levels = {} # Indicate failure to build levels
                         return


        if lowest_floor is None:
             print("Error: Could not determine lowest floor from 'above' predicates.")
             self.floor_levels = {} # Indicate failure to build levels
             return

        # Build the level map starting from the lowest floor
        current_floor = lowest_floor
        level = 1
        while current_floor is not None:
            self.floor_levels[current_floor] = level
            current_floor = floor_above_map.get(current_floor)
            level += 1

        # Basic check if all floors were assigned a level
        if len(self.floor_levels) != len(all_floors):
             print("Warning: Not all floors assigned a level. 'above' predicates may not form a single chain.")
             # Heuristic might be inaccurate for these problems.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all unserved passengers.
        """
        state = node.state

        # If floor levels weren't built correctly, return infinity
        if not self.floor_levels:
             return float('inf')

        # Find the current lift floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        if lift_floor is None:
             # This should not happen in a valid miconic state, but handle defensively
             print("Error: Lift location not found in state.")
             return float('inf')

        lift_level = self.floor_levels.get(lift_floor)
        if lift_level is None:
             # This indicates an inconsistency between state and static facts
             print(f"Error: Lift floor '{lift_floor}' not found in floor levels map.")
             return float('inf')


        total_cost = 0

        # Iterate through all passengers we know about (from destinations)
        for passenger, destin_floor in self.destinations.items():
            # Check if the passenger is already served
            if f"(served {passenger})" in state:
                continue # This passenger is done

            # Get destination level
            destin_level = self.floor_levels.get(destin_floor)
            if destin_level is None:
                 print(f"Error: Destination floor '{destin_floor}' for passenger '{passenger}' not found in floor levels map.")
                 return float('inf')

            # Check if the passenger is waiting at an origin floor
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor = get_parts(fact)[2]
                    break

            if origin_floor:
                # Passenger is waiting at origin
                origin_level = self.floor_levels.get(origin_floor)
                if origin_level is None:
                     print(f"Error: Origin floor '{origin_floor}' for passenger '{passenger}' not found in floor levels map.")
                     return float('inf')

                # Cost = move to origin + board + move to destin + depart
                cost_to_origin = abs(lift_level - origin_level)
                cost_origin_to_destin = abs(origin_level - destin_level)
                passenger_cost = cost_to_origin + 1 + cost_origin_to_destin + 1
                total_cost += passenger_cost

            elif f"(boarded {passenger})" in state:
                # Passenger is already boarded
                # Cost = move to destin + depart
                cost_to_destin = abs(lift_level - destin_level)
                passenger_cost = cost_to_destin + 1
                total_cost += passenger_cost

            # Note: If a passenger is neither 'served', 'origin', nor 'boarded',
            # this heuristic assumes they don't exist or are in an invalid state
            # and skips them. In a valid state, every passenger is either served,
            # waiting at origin, or boarded.

        return total_cost

