import re

def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple (predicate, [args])."""
    # Remove surrounding parentheses and split by whitespace
    parts = fact_str.strip('()').split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the cost to reach the goal by summing the number of unserved
        passengers and the minimum vertical travel distance required for the
        lift to visit all necessary floors (origins for unboarded passengers,
        destinations for boarded passengers).

    Assumptions:
        - Floor names are in the format 'f<number>' and represent a linear
          sequence of floors ordered by the number.
        - Passenger names are in the format 'p<number>'.
        - The goal is to serve a specific set of passengers.
        - The input state and static facts follow the PDDL structure
          represented as frozensets of strings.
        - The initial state and static facts contain enough information to
          determine the floor ordering and passenger destinations.
        - Valid states always contain exactly one (lift-at ?f) fact.
        - Unserved, unboarded goal passengers always have an (origin ?p ?f)
          fact in the state.
        - Unserved, boarded goal passengers always have a (destin ?p ?f)
          fact in the static information.

    Heuristic Initialization:
        - Parses all floor names from the initial state, goal, and static facts.
        - Creates a mapping from floor name string (e.g., 'f1') to an integer
          level (e.g., 1) based on the numerical order of the floor names.
        - Stores the goal facts and static facts for quick lookup during
          heuristic computation.
        - Extracts and stores passenger destinations from static facts.
        - Extracts and stores the set of passengers that are part of the goal.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Check if the current state is a goal state by verifying if all
            goal facts are present in the state. If yes, the heuristic is 0.
        2.  Identify the current floor of the lift by finding the fact
            '(lift-at ?f)' in the state.
        3.  Initialize a counter for unserved goal passengers (`num_unserved`)
            and a set for floors the lift must visit (`floors_to_visit`).
        4.  Iterate through each passenger identified as a goal passenger
            during initialization:
            - If the fact '(served <passenger>)' is not present in the current state:
                - Increment `num_unserved`.
                - Check if the fact '(boarded <passenger>)' is present in the state:
                    - If boarded, the passenger needs to be dropped off. Add their
                      destination floor (looked up from pre-parsed static facts)
                      to `floors_to_visit`.
                    - If not boarded, the passenger needs to be picked up. Find
                      their current origin floor by searching for the fact
                      '(origin <passenger> ?f)' in the current state and add
                      this floor to `floors_to_visit`.
        5.  If `num_unserved` is 0, the heuristic is 0 (redundant check, but safe).
        6.  If `floors_to_visit` is empty (which implies all unserved goal
            passengers are boarded and at their destination floor), the only
            remaining actions are the 'depart' actions. The heuristic is
            simply `num_unserved`.
        7.  If `floors_to_visit` is not empty:
            - Get the integer level corresponding to the current lift floor
              using the pre-calculated floor level mapping.
            - Get the integer levels for all floors in `floors_to_visit`.
            - Determine the minimum (`min_level`) and maximum (`max_level`)
              levels among the current lift floor's level and all levels in
              `floors_to_visit`.
            - The estimated vertical travel component of the heuristic is the
              difference between `max_level` and `min_level`. This represents
              the minimum vertical span the lift must cover to service the
              required floors, starting from its current position.
            - The total heuristic value is the sum of the estimated travel
              component and the `num_unserved`. This combines the cost of
              vertical movement with a lower bound on the number of boarding
              and departing actions.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.floor_levels = self._build_floor_levels(task)
        self.passenger_destins = self._get_passenger_destins()
        self.goal_passengers = self._get_goal_passengers()

    def _build_floor_levels(self, task):
        """Builds a mapping from floor name string to integer level."""
        floor_names = set()
        # Collect all terms that look like floors from initial state, goal, static
        # This is a heuristic way to find all floor objects
        # Consider facts from initial_state, goals, and static
        all_facts = task.initial_state | task.goals | task.static

        for fact_str in all_facts:
             _, args = parse_fact(fact_str)
             for arg in args:
                 # Assuming floors are named f<number>
                 if re.match(r'^f\d+$', arg):
                     floor_names.add(arg)

        # Sort floor names numerically (e.g., f1, f2, f10)
        # Use a robust key function that handles potential errors, though regex should ensure format
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))

        # Create mapping from floor name to level (1-based index)
        floor_levels = {floor_name: i + 1 for i, floor_name in enumerate(sorted_floor_names)}
        return floor_levels

    def _get_passenger_destins(self):
        """Extracts passenger destinations from static facts."""
        passenger_destins = {}
        for fact_str in self.static:
            pred, args = parse_fact(fact_str)
            if pred == 'destin':
                # Fact is (destin ?person ?floor)
                person = args[0]
                floor = args[1]
                passenger_destins[person] = floor
        return passenger_destins

    def _get_goal_passengers(self):
        """Extracts the set of passengers that need to be served from goal facts."""
        goal_passengers = set()
        for fact_str in self.goals:
            pred, args = parse_fact(fact_str)
            if pred == 'served':
                # Fact is (served ?person)
                goal_passengers.add(args[0])
        return goal_passengers


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: A frozenset of strings representing the current state facts.
        @return: An integer heuristic value.
        """
        # 1. Check if goal is reached
        if self.goals <= state:
             return 0

        # 2. Identify current lift floor
        current_lift_floor = None
        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'lift-at':
                current_lift_floor = args[0]
                break

        # Assumption: current_lift_floor is always found in a valid state

        # 3. Initialize counters and sets
        num_unserved = 0
        floors_to_visit = set()

        # 4. Iterate through goal passengers to find unserved and required stops
        for passenger in self.goal_passengers:
            served_fact = f'(served {passenger})'
            if served_fact not in state:
                num_unserved += 1
                boarded_fact = f'(boarded {passenger})'
                if boarded_fact in state:
                    # Passenger is boarded, needs to go to destination
                    dest_floor = self.passenger_destins.get(passenger)
                    # Assumption: dest_floor exists for goal passengers
                    if dest_floor:
                         floors_to_visit.add(dest_floor)
                else:
                    # Passenger is not boarded, needs to be picked up at origin
                    origin_floor = None
                    # Find origin from current state facts
                    for fact_str in state:
                        pred, args = parse_fact(fact_str)
                        if pred == 'origin' and args[0] == passenger:
                            origin_floor = args[1]
                            break
                    # Assumption: origin_floor exists for unserved, unboarded goal passengers
                    if origin_floor:
                         floors_to_visit.add(origin_floor)

        # 5. If no unserved passengers, heuristic is 0 (redundant check, but safe)
        if num_unserved == 0:
            return 0

        # 6. If no floors need visiting (all unserved are boarded and at destination)
        if not floors_to_visit:
            return num_unserved # Only depart actions remaining

        # 7. Calculate travel component
        current_level = self.floor_levels[current_lift_floor]
        visit_levels = {self.floor_levels[f] for f in floors_to_visit}

        # Include the current lift floor in the range calculation
        all_levels_involved = visit_levels | {current_level}

        min_level = min(all_levels_involved)
        max_level = max(all_levels_involved)

        # Estimated travel is the range of floors including the lift's current floor
        estimated_travel = max_level - min_level

        # Total heuristic: travel + actions (at least one per unserved passenger)
        return estimated_travel + num_unserved
