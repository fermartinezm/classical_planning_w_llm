from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment if running in the planner environment

# Define a dummy Heuristic base class if running standalone for testing
# In the actual planner environment, this should be provided.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact string
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

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
    This heuristic estimates the number of actions required to serve all
    passengers by summing the estimated cost for each unserved passenger
    independently. The cost for a passenger includes the necessary board/depart
    actions and the estimated lift movement required to pick them up (if unboarded)
    and transport them to their destination. It assumes that moving the lift
    between any two different floors costs 1 action, provided the 'above'
    predicate allows the move direction.

    # Assumptions
    - The `above` predicate defines the relative height of floors, and the actions
      `up` and `down` allow moving the lift between any two floors `f1` and `f2`
      with cost 1, provided the relevant `above` predicate holds (i.e., `(above f1 f2)`
      for `up` or `(above f2 f1)` for `down`).
    - The cost of each action (board, depart, up, down) is 1.
    - The heuristic calculates the cost for each unserved passenger as:
      - If unboarded: 2 (board + depart) + distance(current_lift_floor, origin) + distance(origin, destination).
      - If boarded: 1 (depart) + distance(current_lift_floor, destination).
    - The distance between any two different floors is 1 move action.
    - The total heuristic is the sum of these individual passenger costs. This is non-admissible
      as it overestimates movement costs when multiple passengers share floors or paths.

    # Heuristic Initialization
    - Extracts the origin and destination floors for each passenger from static facts.
    - Identifies all passengers in the problem.
    - The floor structure and `above` predicates are used implicitly by assuming
      any necessary move between different floors costs 1.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift from the state facts.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through all passengers defined in the problem (extracted during initialization).
    4. For each passenger `p`:
       a. Check if the passenger is currently served by looking for the fact `(served p)` in the state. If this fact exists, the passenger is served, and they contribute 0 to the heuristic cost. Continue to the next passenger.
       b. If the passenger is not served, determine their origin and destination floors (extracted during initialization).
       c. Check if the passenger is currently boarded by looking for the fact `(boarded p)` in the state.
       d. If the passenger is boarded:
          - They need 1 more action: `depart`.
          - The lift needs to move from its current floor to the passenger's destination floor. The cost of this movement is 1 if the current floor is different from the destination floor, and 0 if they are the same.
          - The cost for this passenger is 1 (depart) + movement cost. Add this to the total heuristic cost.
       e. If the passenger is not boarded (and not served, implying they are at their origin floor, which is checked by the presence of `(origin p f)` fact in state):
          - They need 2 more actions: `board` and `depart`.
          - The lift needs to move from its current floor to the passenger's origin floor. The cost is 1 if different, 0 if same.
          - The lift then needs to move from the passenger's origin floor to their destination floor. The cost is 1 if different, 0 if same.
          - The cost for this passenger is 2 (board + depart) + movement cost to origin + movement cost from origin to destination. Add this to the total heuristic cost.
    5. Return the total calculated heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Passenger origin and destination floors.
        - Set of all passengers.
        """
        self.goals = task.goals
        self.static = task.static

        # --- Extract Passenger Origin and Destination ---
        self.origin_map = {}
        self.destin_map = {}
        self.all_passengers = set()

        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "origin", "*", "*"):
                p, f = parts[1], parts[2]
                self.origin_map[p] = f
                self.all_passengers.add(p)
            elif match(fact, "destin", "*", "*"):
                p, f = parts[1], parts[2]
                self.destin_map[p] = f
                self.all_passengers.add(p)

    def get_floor_distance(self, f1, f2):
        """
        Helper to calculate the distance (number of moves) between two floors.
        Assumes direct moves between any two different floors cost 1.
        """
        return 0 if f1 == f2 else 1

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract State Information ---
        current_lift_floor = None
        boarded_passengers_in_state = set()
        served_passengers_in_state = set()
        origin_passengers_in_state = set() # Passengers currently at their origin floor

        # Build sets from state facts for quick lookup
        state_facts_set = set(state)

        for fact in state_facts_set:
            parts = get_parts(fact)
            if match(fact, "lift-at", "*"):
                current_lift_floor = parts[1]
            elif match(fact, "boarded", "*"):
                boarded_passengers_in_state.add(parts[1])
            elif match(fact, "served", "*"):
                served_passengers_in_state.add(parts[1])
            elif match(fact, "origin", "*", "*"):
                 # This fact indicates the passenger is unboarded and at their origin
                 origin_passengers_in_state.add(parts[1])

        # Ensure we found the lift location
        if current_lift_floor is None:
             # This indicates an invalid state representation or problem definition
             # Return a very high heuristic value to prune this state
             return 1000000 # A large number indicating high cost

        # --- Calculate Heuristic ---
        total_cost = 0

        # Iterate through all passengers defined in the problem
        for p in self.all_passengers:
            # Check if the passenger is served in the current state
            if p in served_passengers_in_state:
                # This passenger is served, contributes 0 to heuristic
                continue

            # Passenger is not served. Calculate cost for this passenger.
            p_origin = self.origin_map.get(p)
            p_destin = self.destin_map.get(p)

            # Defensive check: ensure origin and destination are defined for the passenger
            if p_origin is None or p_destin is None:
                 # This passenger cannot be served based on static facts.
                 # This might indicate an invalid problem definition or a passenger
                 # not relevant to the goal (if goals only specify a subset).
                 # Assuming all passengers in self.all_passengers must be served.
                 # If they cannot be served, the problem is unsolvable from this state.
                 # Return a very high heuristic value.
                 return 1000000

            if p in boarded_passengers_in_state:
                # Passenger is boarded, needs to depart at destination
                # Cost = 1 (depart action) + movement from current lift floor to destination
                cost_p = 1 + self.get_floor_distance(current_lift_floor, p_destin)
                total_cost += cost_p
            elif p in origin_passengers_in_state:
                 # Passenger is unboarded and at their origin
                 # Cost = 2 (board + depart actions) + movement from current lift floor to origin
                 # + movement from origin to destination
                 cost_p = 2 + self.get_floor_distance(current_lift_floor, p_origin) + self.get_floor_distance(p_origin, p_destin)
                 total_cost += cost_p
            # else:
            #     # Passenger is not served, not boarded, and not at their origin?
            #     # This state should not be reachable in a valid execution sequence
            #     # from the initial state, assuming initial state is valid.
            #     # For robustness, could return a high cost or ignore. Ignoring for now
            #     # assumes valid states only have passengers in served, boarded, or origin predicates.
            #     pass

        return total_cost
