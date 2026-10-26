from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and parentheses
    fact = fact.strip()
    if fact.startswith('(') and fact.endswith(')'):
        fact = fact[1:-1]
    return fact.split()

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
    This heuristic estimates the number of actions (move, board, depart) required
    to transport all unserved passengers to their destinations. It calculates the
    cost for each unserved passenger independently and sums these costs. The cost
    for a passenger includes the estimated lift movement to pick them up (if waiting)
    and drop them off, plus the board and depart actions.

    # Assumptions
    - Floors are ordered numerically (e.g., f1, f2, f10). The heuristic parses
      floor names assuming this format and sorts them accordingly.
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of a board action is 1.
    - The cost of a depart action is 1.
    - The heuristic assumes a simplified model where each passenger's transport
      cost is calculated independently, ignoring potential efficiencies from
      batching passengers in the lift. This makes the heuristic non-admissible
      but potentially effective for greedy search.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Collects all unique floor names from static facts and the initial state.
    - Sorts the floor names numerically to establish the floor order.
    - Creates a mapping from each floor name to its numerical index in the sorted list.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information from the State:
       - Identify the current floor of the lift.
       - Identify which passengers are waiting at their origin floors.
       - Identify which passengers are currently boarded in the lift.
       - Identify which passengers have already been served.

    2. Identify Unserved Passengers:
       - Determine the set of all passengers (from the destinations extracted during initialization).
       - Subtract the set of served passengers to get the set of unserved passengers.

    3. Check for Goal State:
       - If the set of unserved passengers is empty, the goal is reached, and the heuristic value is 0.

    4. Calculate Heuristic for Unserved Passengers:
       - Initialize the total heuristic value `h` to 0.
       - Get the index of the current lift floor using the floor-to-index mapping.
       - Iterate through each unserved passenger `p`:
         - If `p` is waiting at their origin floor `origin_f`:
           - Get the destination floor `destin_f` for `p` (from initialization data).
           - Get the indices for `origin_f` and `destin_f`.
           - Estimate the movement cost for `p`: `abs(current_lift_index - origin_index) + abs(origin_index - destin_index)`. This represents moving to the origin floor and then to the destination floor.
           - Add the movement cost to `h`.
           - Add the action cost for `p`: 2 (1 for board, 1 for depart). Add this to `h`.
         - If `p` is currently boarded in the lift:
           - Get the destination floor `destin_f` for `p`.
           - Get the index for `destin_f`.
           - Estimate the movement cost for `p`: `abs(current_lift_index - destin_index)`. This represents moving directly to the destination floor.
           - Add the movement cost to `h`.
           - Add the action cost for `p`: 1 (for depart). Add this to `h`.
         - (Passengers who are unserved but neither waiting nor boarded are not expected in valid states and are implicitly ignored by this logic).

    5. Return the Total Heuristic Value:
       - The final value of `h` is the estimated total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Extract destination floors for each passenger from static facts
        self.destinations = {}
        all_floors = set()

        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts[0] == "destin":
                 # Store destination for each passenger
                 self.destinations[parts[1]] = parts[2]
                 # Collect all floors mentioned in destinations
                 all_floors.add(parts[2])
             elif parts[0] == "above":
                 # Collect all floors mentioned in above facts
                 all_floors.add(parts[1])
                 all_floors.add(parts[2])

        # Also get floors from initial state (lift-at, origin)
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == "lift-at":
                 all_floors.add(parts[1])
             elif parts[0] == "origin":
                 # Collect all floors mentioned in origins
                 all_floors.add(parts[2])

        # Sort floors numerically based on the number part (e.g., f1, f2, f10)
        # This assumes floor names are consistently formatted like 'f<number>'
        def floor_sort_key(floor_name):
            match = re.match(r'f(\d+)', floor_name)
            if match:
                return int(match.group(1))
            # Fallback for non-standard names, though problem likely uses f<number>
            return floor_name

        self.floor_list = sorted(list(all_floors), key=floor_sort_key)
        self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_list)}

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify current lift location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid miconic state
             return float('inf')

        current_lift_index = self.floor_to_index[current_lift_floor]

        # 2. Identify state of passengers
        waiting_passengers = {} # passenger -> origin_floor
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                waiting_passengers[parts[1]] = parts[2]
            elif parts[0] == "boarded":
                boarded_passengers.add(parts[1])
            elif parts[0] == "served":
                served_passengers.add(parts[1])

        # 3. Identify unserved passengers (those with destinations who are not served)
        # We iterate through passengers found in destinations during init
        unserved_passengers = {p for p in self.destinations.keys() if p not in served_passengers}

        # 4. Check if goal is reached
        if not unserved_passengers:
            return 0 # All passengers with destinations are served

        # 5. Calculate Heuristic for Unserved Passengers
        total_cost = 0

        for p in unserved_passengers:
            # A passenger must be either waiting or boarded if unserved and part of the problem
            # (i.e., has a destination). If they are neither, something is wrong with the state
            # or problem definition, but we'll only calculate cost if they are in a known state.
            if p in waiting_passengers:
                origin_f = waiting_passengers[p]
                destin_f = self.destinations[p]

                # Ensure floors exist in our mapping (robustness)
                if origin_f not in self.floor_to_index or destin_f not in self.floor_to_index:
                     # This shouldn't happen in valid problems
                     return float('inf') # Indicate invalid state

                origin_index = self.floor_to_index[origin_f]
                destin_index = self.floor_to_index[destin_f]

                # Movement cost: current -> origin -> destination
                # This is a simple estimate ignoring intermediate stops for other passengers
                movement_cost = abs(current_lift_index - origin_index) + abs(origin_index - destin_index)
                # Action cost: board + depart
                action_cost = 2
                total_cost += movement_cost + action_cost

            elif p in boarded_passengers:
                destin_f = self.destinations[p]

                # Ensure floor exists in our mapping (robustness)
                if destin_f not in self.floor_to_index:
                     # This shouldn't happen in valid problems
                     return float('inf') # Indicate invalid state

                destin_index = self.floor_to_index[destin_f]

                # Movement cost: current -> destination
                # This is a simple estimate ignoring intermediate stops for other passengers
                movement_cost = abs(current_lift_index - destin_index)
                # Action cost: depart
                action_cost = 1
                total_cost += movement_cost + action_cost

            # If a passenger is in self.destinations but not in waiting_passengers or boarded_passengers
            # in the current state, they are effectively "lost" from the perspective of this heuristic.
            # This shouldn't happen in a reachable state from a valid initial state and operators.
            # We don't add cost for them in this case.

        return total_cost
