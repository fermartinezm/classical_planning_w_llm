from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements and boarding/departing actions for each passenger.

    # Assumptions:
    - The lift can move between floors and serve passengers one by one.
    - Each passenger requires a 'board' and 'depart' action.
    - The heuristic assumes the lift starts at its current position and moves optimally.

    # Heuristic Initialization
    - Extracts the floor hierarchy from static facts to determine the distance between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current position of the lift.
    2. For each passenger, determine if they are served, boarded, their origin, and destination.
    3. For each unserved passenger:
       a. If not boarded, calculate the distance from the lift's current position to their origin, add boarding action.
       b. Calculate the distance from origin to destination, add departing action.
       c. If boarded, calculate the distance from the lift's current position to their destination, add departing action.
    4. Sum all the calculated actions for an estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting floor hierarchy from static facts."""
        self.floor_level = {}
        # Extract floor levels from task.objects
        for obj in task.objects:
            if obj[0] == 'floor':
                f = obj[1]
                # Extract number from floor name (assuming f1, f2, etc.)
                match = re.match(r'f(\d+)', f)
                if match:
                    level = int(match.group(1))
                    self.floor_level[f] = level

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        lift_floor = None
        for fact in state:
            if match(fact, 'lift-at', '*'):
                lift_floor = get_parts(fact)[1]
                break

        if lift_floor is None:
            # No lift position found, which is an error
            return 0

        total_cost = 0

        # Extract all passengers
        passengers = set()
        for fact in state:
            if match(fact, 'origin', '*', '*'):
                p = get_parts(fact)[1]
                passengers.add(p)

        for p in passengers:
            # Check if passenger is served
            is_served = any(match(f, 'served', p) for f in state)
            if is_served:
                continue

            # Check if passenger is boarded
            is_boarded = any(match(f, 'boarded', p) for f in state)
            if not is_boarded:
                # Find origin
                origin = None
                for fact in state:
                    if match(fact, 'origin', p, '*'):
                        origin = get_parts(fact)[2]
                        break
                if origin is None:
                    continue  # error

                # Find destination
                dest = None
                for fact in state:
                    if match(fact, 'destin', p, '*'):
                        dest = get_parts(fact)[2]
                        break
                if dest is None:
                    continue  # error

                # Calculate distance from lift to origin
                if origin in self.floor_level and lift_floor in self.floor_level:
                    distance_origin = abs(self.floor_level[origin] - self.floor_level[lift_floor])
                else:
                    distance_origin = 0

                # Add move and board actions
                total_cost += distance_origin + 1  # move and board

                # Calculate distance from origin to destination
                if origin in self.floor_level and dest in self.floor_level:
                    distance_dest = abs(self.floor_level[origin] - self.floor_level[dest])
                else:
                    distance_dest = 0

                # Add move and depart actions
                total_cost += distance_dest + 1  # move and depart
            else:
                # Passenger is boarded, find destination
                dest = None
                for fact in state:
                    if match(fact, 'destin', p, '*'):
                        dest = get_parts(fact)[2]
                        break
                if dest is None:
                    continue  # error

                # Calculate distance from lift to destination
                if lift_floor in self.floor_level and dest in self.floor_level:
                    distance_dest = abs(self.floor_level[lift_floor] - self.floor_level[dest])
                else:
                    distance_dest = 0

                # Add move and depart actions
                total_cost += distance_dest + 1  # move and depart

        return total_cost

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))
