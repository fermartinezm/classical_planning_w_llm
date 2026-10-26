from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re # Import regex for parsing floor names

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check: just zip and compare, fnmatch handles length differences gracefully
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the number of necessary board and depart actions and adds an estimate
    of the lift movement cost.

    # Assumptions
    - Floors are numbered sequentially (f1, f2, f3, ...).
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding is 1.
    - The cost of departing is 1.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the goal state.
    - Creates a mapping from floor names (e.g., 'f1', 'f10') to integer floor numbers.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify all passengers who have not yet been served (i.e., the fact `(served p)` is not in the state).
    2. Count the number of 'board' actions needed: This is the number of unserved passengers currently waiting at their origin floor (`(origin p f)` is in the state). Each such passenger needs one 'board' action.
    3. Count the number of 'depart' actions needed: This is the total number of unserved passengers (those waiting *or* boarded). Each unserved passenger eventually needs one 'depart' action at their destination.
    4. Estimate the lift movement cost: The lift must visit its current floor, the origin floors of all waiting passengers, and the destination floors of all unserved passengers (both currently waiting and currently boarded). The minimum vertical travel required to visit a set of floors on a line is the range between the lowest and highest floor in that set. We use this range as the movement cost estimate.
    5. The total heuristic value is the sum of the board actions needed, the depart actions needed, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and floor mapping.
        """
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static

        # Store goal locations for each passenger.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "served":
                 # Find the destination for this passenger from static facts
                 passenger = args[0]
                 destin_fact = next((fact for fact in self.static_facts if match(fact, "destin", passenger, "*")), None)
                 if destin_fact:
                     _, p, floor = get_parts(destin_fact)
                     self.goal_locations[p] = floor
                 # If destin fact not found in static, it might be in initial state
                 # (though PDDL convention usually puts static facts in :static)
                 # For robustness, could also check initial state or assume all destins are static.
                 # Based on examples, destin is static.

        # Build floor name to integer mapping.
        # Collect all floor names from static facts (above) and initial state (lift-at, origin, destin).
        floor_names = set()
        for fact in self.static_facts | task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["above", "lift-at", "origin", "destin"]:
                 for part in parts[1:]:
                     # Floor names start with 'f'
                     if isinstance(part, str) and part.startswith('f'):
                         floor_names.add(part)

        # Sort floor names numerically (e.g., f1, f2, ..., f10, f11)
        # Use regex to extract the number part
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(re.findall(r'\d+', f)[0]))

        self.floor_name_to_int = {name: i + 1 for i, name in enumerate(sorted_floor_names)}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve
        all remaining passengers.
        """
        state = node.state

        # 1. Identify unserved passengers
        all_passengers = set(self.goal_locations.keys())
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = all_passengers - served_passengers

        # If all passengers are served, the heuristic is 0.
        if not unserved_passengers:
            return 0

        # 2. Count board actions needed
        board_actions_needed = 0
        waiting_passengers = set()
        for passenger in unserved_passengers:
            # Check if the passenger is waiting at their origin floor
            origin_fact = next((fact for fact in state if match(fact, "origin", passenger, "*")), None)
            if origin_fact:
                board_actions_needed += 1
                waiting_passengers.add(passenger)

        # 3. Count depart actions needed
        depart_actions_needed = len(unserved_passengers) # Each unserved passenger needs one depart action

        # 4. Estimate lift movement cost
        # Find the lift's current floor
        current_lift_floor_name = next(get_parts(fact)[1] for fact in state if match(fact, "lift-at", "*"))
        current_lift_floor_int = self.floor_name_to_int[current_lift_floor_name]

        # Collect all relevant floor numbers: lift's current floor, origins of waiting, destins of unserved
        relevant_floor_numbers = {current_lift_floor_int}

        for passenger in waiting_passengers:
            # Add origin floor
            origin_fact = next(fact for fact in state if match(fact, "origin", passenger, "*"))
            origin_floor_name = get_parts(origin_fact)[2]
            relevant_floor_numbers.add(self.floor_name_to_int[origin_floor_name])

            # Add destination floor
            destin_floor_name = self.goal_locations[passenger]
            relevant_floor_numbers.add(self.floor_name_to_int[destin_floor_name])

        # Add destination floors for boarded passengers (who are also unserved)
        boarded_unserved_passengers = unserved_passengers - waiting_passengers
        for passenger in boarded_unserved_passengers:
             # Add destination floor
            destin_floor_name = self.goal_locations[passenger]
            relevant_floor_numbers.add(self.floor_name_to_int[destin_floor_name])


        # The movement cost is the range of these relevant floors
        min_floor = min(relevant_floor_numbers)
        max_floor = max(relevant_floor_numbers)
        movement_cost = max_floor - min_floor

        # Total heuristic is the sum of action counts and movement estimate
        total_cost = board_actions_needed + depart_actions_needed + movement_cost

        return total_cost

