# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle standard PDDL fact format
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        # Split by space, assuming arguments do not contain spaces themselves
        return fact_str[1:-1].split()
    # This case should ideally not be reached for valid PDDL facts in state/static
    # Return the string itself in a list as a fallback, though it might cause errors later
    # if not handled by the caller. For standard PDDL, the split() above is sufficient.
    return [fact_str]

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions for unserved passengers
    and adds an estimate of the necessary lift movement actions.

    # Assumptions
    - Each unserved passenger needs one board action (if waiting) and one depart action.
    - Lift movement cost is estimated based on the vertical distance needed to reach
      and traverse the range of floors relevant to unserved passengers.
    - Actions have a cost of 1.

    # Heuristic Initialization
    - Build a mapping from floor names to numerical indices based on the `above` static facts.
    - Store the destination floor for each passenger from the `destin` static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the lift's current floor.
    2. Identify all unserved passengers (those not in a `(served ?p)` fact).
    3. If no passengers are unserved, the heuristic is 0.
    4. Count the number of passengers currently waiting at their origin floor (`N_waiting`).
    5. The estimated number of board/depart actions is `N_waiting` (for boarding) + `N_unserved` (for departing).
    6. Identify all floors that are relevant to the remaining task:
       - Origin floors of waiting passengers.
       - Destination floors of boarded passengers.
       - Destination floors of waiting passengers (as they will eventually be boarded and need dropping off).
    7. If there are no relevant floors (which should only happen if there are no unserved passengers, handled in step 3), the move cost is 0.
    8. Otherwise, find the minimum and maximum floor indices among the relevant floors.
    9. Calculate the estimated move cost:
       - The lift must travel from its current floor to the closest end of the range [min_relevant_idx, max_relevant_idx], and then traverse the entire span (max_relevant_idx - min_relevant_idx).
       - Move cost = `min(abs(current_idx - min_relevant_idx), abs(current_idx - max_relevant_idx)) + (max_relevant_idx - min_relevant_idx)`.
    10. The total heuristic value is the sum of the board/depart action count and the estimated move cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Build floor index map
        all_floors = set()
        # Collect floors from static facts (above, destin)
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                all_floors.add(parts[1])
                all_floors.add(parts[2])
            if parts and parts[0] == "destin" and len(parts) == 3:
                 all_floors.add(parts[2])
        # Collect floors from initial state facts (lift-at, origin)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "lift-at" and len(parts) == 2:
                  all_floors.add(parts[1])
             if parts and parts[0] == "origin" and len(parts) == 3:
                  all_floors.add(parts[2])

        # Count how many floors each floor is above
        floor_above_counts = {f: 0 for f in all_floors}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                if parts[1] in floor_above_counts: # Ensure floor is in our set
                    floor_above_counts[parts[1]] += 1

        # Sort floors by the count (ascending) to get order from lowest to highest
        # Handle cases where all_floors might be empty
        if not all_floors:
             floors_sorted = []
        else:
             # Use floor_above_counts.get(f, 0) to handle any floor in all_floors
             # that might not have appeared as the first argument in an 'above' fact
             floors_sorted = sorted(all_floors, key=lambda f: floor_above_counts.get(f, 0))

        # Assign numerical index to each floor
        self.floor_indices = {f: i for i, f in enumerate(floors_sorted)}

        # Store passenger destinations
        self.passenger_destins = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3: # Ensure fact is (destin passenger floor)
                self.passenger_destins[parts[1]] = parts[2] # passenger -> destination_floor

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify the lift's current floor.
        lift_at_fact = next((fact for fact in state if match(fact, "lift-at", "*")), None)
        if lift_at_fact is None:
             # Should not happen in a valid miconic state
             return float('inf')

        current_floor = get_parts(lift_at_fact)[1]
        current_idx = self.floor_indices.get(current_floor)
        if current_idx is None:
             # Should not happen if floor indexing is correct and covers all floors in states
             # This might indicate a floor in the state wasn't in static/initial state during init
             return float('inf')

        # 2. Identify all unserved passengers.
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*") and len(get_parts(fact)) > 1}

        # Collect all passengers mentioned in the current state or static destinations
        all_passengers = set(self.passenger_destins.keys())
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ["origin", "boarded", "served"] and len(parts) > 1:
                  all_passengers.add(parts[1])

        unserved_passengers = all_passengers - served_passengers

        # 3. If no passengers are unserved, the heuristic is 0.
        if not unserved_passengers:
            return 0

        # 4. Count waiting passengers.
        waiting_passengers = {get_parts(fact)[1] for fact in state if match(fact, "origin", "*", "*") and len(get_parts(fact)) > 1}
        N_waiting = len(waiting_passengers)
        N_unserved = len(unserved_passengers)

        # 5. Estimated board/depart actions.
        # Each waiting passenger needs a board action.
        # Each unserved passenger needs a depart action eventually.
        action_cost = N_waiting + N_unserved

        # 6. Identify relevant floors.
        relevant_floors = set()
        # Origin floors of waiting passengers
        for fact in state:
            if match(fact, "origin", "*", "*") and len(get_parts(fact)) > 2:
                relevant_floors.add(get_parts(fact)[2])
        # Destination floors of boarded passengers
        for fact in state:
            if match(fact, "boarded", "*") and len(get_parts(fact)) > 1:
                 p = get_parts(fact)[1]
                 if p in self.passenger_destins:
                    relevant_floors.add(self.passenger_destins[p])
        # Destination floors of waiting passengers
        for p in waiting_passengers:
             if p in self.passenger_destins:
                  relevant_floors.add(self.passenger_destins[p])

        # Filter out any relevant floors that weren't indexed (shouldn't happen with robust floor collection)
        relevant_floors = {f for f in relevant_floors if f in self.floor_indices}

        # 7. Calculate estimated move cost.
        move_cost = 0
        if relevant_floors:
            relevant_indices = {self.floor_indices[f] for f in relevant_floors}
            min_idx = min(relevant_indices)
            max_idx = max(relevant_indices)

            # Move cost estimate: travel to the closest end of the range and traverse the span.
            dist_to_min = abs(current_idx - min_idx)
            dist_to_max = abs(current_idx - max_idx)
            span = max_idx - min_idx

            move_cost = min(dist_to_min, dist_to_max) + span

        # 10. Total heuristic value.
        return action_cost + move_cost
