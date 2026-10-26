import re

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the remaining cost by summing the number of unserved passengers
        (representing the board/depart actions needed) and the estimated lift
        movement cost. The movement cost is estimated as the vertical distance
        between the minimum and maximum floor levels that the lift must visit
        to pick up waiting passengers and drop off boarded passengers, including
        the current lift floor.

    Assumptions:
        - The PDDL instance is valid and solvable.
        - Floor names are structured such that sorting them numerically by the
          number part corresponds to their vertical order (e.g., f1 < f2 < f3).
        - Every passenger has a defined destination in the static facts.
        - The state always contains exactly one (lift-at ?f) fact.
        - The goal is defined as a conjunction of (served ?p) facts for all
          passengers that need to be served.

    Heuristic Initialization:
        The constructor parses the static facts and initial state to determine
        the floor order and assign a numerical level to each floor. It also
        extracts the destination floor for each passenger from the static facts
        and identifies the set of passengers that need to be served based on the goal.
        1. Identify all floor objects mentioned in static facts and initial state.
        2. Sort floor objects based on the numerical part of their name (e.g., f1 < f2).
        3. Create a dictionary mapping each sorted floor name to its level (1-based index).
        4. Create a dictionary mapping each passenger name to their destination floor
           from the (destin ?p ?f) static facts.
        5. Identify the set of passengers that appear in the goal facts (those that need to be served).

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Check if the state is a goal state (all goal passengers have the (served ?p) fact). If yes, return 0.
        2. Find the current floor of the lift by looking for the (lift-at ?f) fact and get its level.
        3. Identify unserved goal passengers and categorize them based on the state:
           - 'waiting': passengers with an (origin ?p ?f) fact. Store their origin floor.
           - 'boarded': passengers with a (boarded ?p) fact.
           - 'served': passengers with a (served ?p) fact. (These are excluded from unserved).
        4. Calculate the non-movement cost: This is the sum of the number of waiting
           passengers and the number of boarded passengers among the unserved goal passengers.
           Each represents a board or depart action needed for that passenger.
        5. Identify the set of 'required floors' that the lift must visit:
           - The origin floor for every waiting unserved passenger.
           - The destination floor for every boarded unserved passenger (using the pre-parsed destinations).
        6. If there are no unserved passengers (already handled by step 1), the heuristic is 0.
        7. If there are unserved passengers but no required floors (i.e., all unserved
           passengers are either waiting at the current floor or boarded with the
           current floor as destination), the movement cost is 0. The heuristic is
           just the non-movement cost.
        8. If there are required floors, determine the set of floors the lift must
           consider visiting: the current lift floor plus all required floors.
        9. Find the minimum and maximum floor levels among this set of floors.
        10. Calculate the estimated movement cost: This is the difference between the
            maximum and minimum levels.
        11. The total heuristic value is the sum of the non-movement cost and the
            estimated movement cost.
    """

    def __init__(self, task):
        # Pre-process static information

        # 1. Parse floor levels
        # Collect all floor objects from static facts and initial state
        all_objects = set()
        # Include objects from initial state and static facts
        for fact_str in task.initial_state | task.static:
             parts = fact_str[1:-1].split()
             if len(parts) > 1:
                 # Objects are typically arguments
                 for obj in parts[1:]:
                     all_objects.add(obj)

        # Filter for floors (assuming 'f' prefix followed by number) and sort numerically
        floor_names = sorted([obj for obj in all_objects if obj.startswith('f') and obj[1:].isdigit()],
                             key=lambda f: int(f[1:]))

        # Assign 1-based level to each floor
        self.floor_levels = {floor_name: level + 1 for level, floor_name in enumerate(floor_names)}

        # 2. Parse passenger destinations from static facts
        self.destinations = {}
        for fact_str in task.static:
            if fact_str.startswith('(destin '):
                parts = fact_str[1:-1].split()
                # parts should be ['destin', passenger_name, floor_name]
                if len(parts) == 3:
                    passenger = parts[1]
                    destination_floor = parts[2]
                    self.destinations[passenger] = destination_floor

        # 3. Identify goal passengers from goal facts
        self.goal_passengers = set()
        for goal_fact in task.goals:
             if goal_fact.startswith('(served '):
                 parts = goal_fact[1:-1].split()
                 if len(parts) == 2:
                     self.goal_passengers.add(parts[1])

    def __call__(self, state):
        # Check for goal state first
        if self.is_goal(state):
             return 0

        current_lift_floor = None
        waiting_passengers_dict = {} # {passenger: origin_floor} for goal passengers
        boarded_passengers_set = set() # {passenger} for goal passengers
        served_passengers_set = set() # {passenger} for goal passengers

        # Parse state to find lift position and status of goal passengers
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                current_lift_floor = fact_str[1:-1].split()[1]
            elif fact_str.startswith('(origin '):
                parts = fact_str[1:-1].split()
                if len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    if passenger in self.goal_passengers:
                         waiting_passengers_dict[passenger] = floor
            elif fact_str.startswith('(boarded '):
                parts = fact_str[1:-1].split()
                if len(parts) == 2:
                    passenger = parts[1]
                    if passenger in self.goal_passengers:
                         boarded_passengers_set.add(passenger)
            elif fact_str.startswith('(served '):
                 parts = fact_str[1:-1].split()
                 if len(parts) == 2:
                     passenger = parts[1]
                     if passenger in self.goal_passengers:
                         served_passengers_set.add(passenger)

        # Filter out passengers who are already served from waiting/boarded sets
        # (This shouldn't strictly be necessary if state is consistent, but adds robustness)
        waiting_passengers_dict = {p: f for p, f in waiting_passengers_dict.items() if p not in served_passengers_set}
        boarded_passengers_set = {p for p in boarded_passengers_set if p not in served_passengers_set}

        # If there are no unserved goal passengers, heuristic is 0 (already checked by is_goal, but safe)
        if not waiting_passengers_dict and not boarded_passengers_set:
             return 0

        # Calculate non-movement cost: sum of board/depart actions needed for unserved passengers
        # Each waiting passenger needs 1 board + 1 depart (conceptually)
        # Each boarded passenger needs 1 depart (conceptually)
        # A simple count of remaining "stages" (waiting -> boarded -> served)
        non_movement_cost = len(waiting_passengers_dict) + len(boarded_passengers_set)

        # Identify required floors: origin floors of waiting + destination floors of boarded
        required_floors = set()
        required_floors.update(waiting_passengers_dict.values())
        for passenger in boarded_passengers_set:
            # Assuming passenger is in self.destinations if they are a goal passenger
            required_floors.add(self.destinations[passenger])

        # Calculate movement cost
        movement_cost = 0
        if required_floors:
            # Floors whose levels determine the span the lift must cover
            floors_to_consider_for_span = required_floors | {current_lift_floor}

            # Get levels of these floors
            levels_to_consider = {self.floor_levels[f] for f in floors_to_consider_for_span}

            # Movement cost is the vertical span
            min_level = min(levels_to_consider)
            max_level = max(levels_to_consider)
            movement_cost = max_level - min_level
        # If required_floors is empty, movement_cost remains 0, which is correct
        # (The only remaining actions are departs at the current floor, covered by non_movement_cost)


        # Total heuristic value
        heuristic_value = non_movement_cost + movement_cost

        return heuristic_value

    def is_goal(self, state):
        """Helper to check if the state is a goal state."""
        # Check if all goal passengers have the (served p) fact in the state
        for passenger in self.goal_passengers:
            if f'(served {passenger})' not in state:
                return False
        return True
