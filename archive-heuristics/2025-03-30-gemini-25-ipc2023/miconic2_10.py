import itertools
from fnmatch import fnmatch
# Assume Heuristic base class is available, e.g.:
# from heuristics.heuristic_base import Heuristic
# Define a dummy if running standalone or if the path is different:
try:
    # This path might need adjustment depending on the planner's structure
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Defining a dummy base class.")
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError


def get_parts(fact):
    """
    Extract the components of a PDDL fact string.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    Handles facts like '()' or '(pred)' gracefully.
    """
    # Remove leading '(' and trailing ')'
    if len(fact) >= 2 and fact.startswith("(") and fact.endswith(")"):
        content = fact[1:-1]
        if not content: # Handles "()"
            return []
        return content.split()
    else:
        # Return as is or raise error if format is unexpected
        # For robustness, maybe return empty list or the fact itself?
        # Let's assume facts are well-formed PDDL strings.
        # If fact doesn't match expected format, splitting might fail or be incorrect.
        # Returning empty list might be safer than erroring.
        print(f"Warning: Unexpected fact format encountered: {fact}")
        return []


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern. Wildcards (*) are allowed.
    Uses get_parts to parse the fact string.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the pattern length
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (Elevator) domain.

    # Summary
    This heuristic estimates the remaining cost to serve all passengers in the
    Miconic elevator domain. It is designed for use with greedy best-first search,
    prioritizing informativeness over admissibility. The heuristic value combines
    an estimate of the necessary 'board' and 'depart' actions with an estimate
    of the minimum vertical distance the lift must travel.

    # Assumptions
    - The PDDL domain uses `(above f1 f2)` to mean floor `f1` is somewhere above
      floor `f2`, implying a total ordering of floors (like floors in a building).
    - The cost of atomic actions (`up`, `down`, `board`, `depart`) is 1.
    - The goal is achieved when all passengers specified in the static `destin`
      predicates are in the `(served p)` state.
    - The input task provides a consistent definition of floors and their relations
      if multiple floors exist.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the static information from the task.
    - It extracts all passenger destination goals `(destin p f)` into `self.destinations`.
    - It identifies the complete set of passengers involved (`self.all_passengers`).
    - It parses the static `(above f1 f2)` facts to determine the 'level' of each floor.
      The level of a floor `f` is defined as the count of other floors `f_other` such
      that `(above f f_other)` holds. This assigns level 0 to the lowest floor in a
      linear hierarchy. The levels are stored in `self.floor_levels`.
    - Handles edge cases like problems with only one floor or missing `above` predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse State:** The `__call__` method receives a search node containing the
        current state. It parses the state facts to identify:
        - The current floor of the lift `(lift-at ?f)`.
        - Passengers waiting at their origin floors `(origin ?p ?f)`.
        - Passengers currently inside the lift `(boarded ?p)`.
        - Passengers already delivered to their destination `(served ?p)`.
    2.  **Identify Unserved Passengers:** Determine the set of passengers who still
        need to reach their destination (`unserved_passengers`). If this set is empty,
        the goal state is reached, and the heuristic returns 0.
    3.  **Calculate Action Cost (`h_actions`):** Estimate the minimum number of
        `board` and `depart` actions remaining:
        - Add 1 for every passenger currently waiting (`origin ?p ?f`), as they need to `board`.
        - Add 1 for every `unserved_passenger`, as they eventually need to `depart`.
    4.  **Calculate Movement Cost (`h_moves`):** Estimate the minimum lift travel distance:
        - Identify all "target" floors the lift needs to visit:
            - Origin floors of waiting passengers (for pickup).
            - Destination floors of boarded passengers (for drop-off).
            - Destination floors of waiting passengers (for eventual drop-off after pickup).
        - If there are no target floors (no passengers need service), `h_moves` is 0.
        - Otherwise, consider the set of all target floors plus the lift's current floor.
        - Using the precomputed `self.floor_levels`, find the minimum (`min_level`) and
          maximum (`max_level`) floor levels within this combined set of floors.
        - `h_moves` is estimated as `max_level - min_level`. This represents the minimum
          vertical range the lift must cover.
        - Includes fallback logic if floor levels could not be determined during initialization.
    5.  **Combine Costs:** The final heuristic value is `h = h_actions + h_moves`.
        This value estimates the total remaining actions (operational + movement).
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class if necessary
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract passenger destinations and all passengers
        self.destinations = {}
        self.all_passengers = set()
        for fact in static_facts:
            # Use match for robust parsing against pattern "(destin ?p ?f)"
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                # Basic validation of parsed parts
                if len(parts) == 3:
                    passenger = parts[1]
                    floor = parts[2]
                    self.destinations[passenger] = floor
                    self.all_passengers.add(passenger)

        # 2. Determine floor levels from 'above' relations
        self.floor_levels = {}
        all_floors_from_above = set()
        above_relations = []
        for fact in static_facts:
            # Use match for robust parsing against pattern "(above ?f1 ?f2)"
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                # Basic validation of parsed parts
                if len(parts) == 3:
                    f_above = parts[1]
                    f_below = parts[2]
                    all_floors_from_above.add(f_above)
                    all_floors_from_above.add(f_below)
                    above_relations.append((f_above, f_below))

        if all_floors_from_above:
            # Calculate level based on how many distinct floors are below a given floor
            # Assumes 'above' implies a total order, level = count of floors below.
            for f in all_floors_from_above:
                # Count floors f_other such that (above f f_other) exists
                count_below = sum(1 for fa, fb in above_relations if fa == f)
                self.floor_levels[f] = count_below
        else:
            # Handle case with no 'above' facts (e.g., single floor problem)
            # Collect all floors mentioned anywhere in init or static facts
            all_floors_mentioned = set()
            facts_to_check = task.initial_state.union(static_facts)
            for fact in facts_to_check:
                 parts = get_parts(fact)
                 if not parts: continue
                 pred = parts[0]
                 # Check predicates known to involve floors as the last argument
                 if pred in ["origin", "destin", "lift-at"]:
                     if len(parts) > 1:
                         all_floors_mentioned.add(parts[-1])

            if len(all_floors_mentioned) == 1:
                # If only one distinct floor is ever mentioned, assign level 0
                self.floor_levels[list(all_floors_mentioned)[0]] = 0
            elif len(all_floors_mentioned) > 1:
                # Multiple floors exist but no 'above' info - levels are unknown.
                print("Warning: Multiple floors exist but 'above' relations are missing. Floor levels undefined.")
                # self.floor_levels remains empty; fallback logic in __call__ will be used.


    def __call__(self, node):
        state = node.state

        # Find served passengers from the current state
        served_passengers = set()
        for fact in state:
             # Use match for robust parsing against pattern "(served ?p)"
             if match(fact, "served", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2: # Ensure structure (served ?p)
                     served_passengers.add(parts[1])

        # Determine unserved passengers based on initial set and current served set
        unserved_passengers = self.all_passengers - served_passengers

        # If all passengers initially defined are now served, goal reached
        if not unserved_passengers:
            return 0

        # Parse current state details: lift location, waiting, boarded passengers
        lift_floor = None
        waiting_passengers = {} # Map: passenger -> origin_floor
        boarded_passengers = set() # Set: passengers currently in lift

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip if parsing failed or fact was empty

            pred = parts[0]
            # Use match for robust parsing and checking predicate names and arity
            if match(fact, "lift-at", "*"): # Checks predicate "lift-at" and arity 1 (lift-at ?f)
                lift_floor = parts[1]
            elif match(fact, "origin", "*", "*"): # Checks predicate "origin" and arity 2 (origin ?p ?f)
                p, f = parts[1], parts[2]
                # Consider only passengers who are not yet served
                if p in unserved_passengers:
                    waiting_passengers[p] = f
            elif match(fact, "boarded", "*"): # Checks predicate "boarded" and arity 1 (boarded ?p)
                p = parts[1]
                # Consider only passengers who are not yet served
                if p in unserved_passengers:
                    boarded_passengers.add(p)

        # Safety check: Lift location must be known in a valid, non-goal state
        if lift_floor is None:
            # This indicates a potentially inconsistent state or problem definition
            print(f"Critical Warning: Lift location ('lift-at') not found in non-goal state! State: {state}")
            # Fallback: Estimate based only on actions needed, add a penalty for uncertainty
            num_board_needed = len(waiting_passengers)
            num_depart_needed = len(unserved_passengers)
            # Return a high value, possibly related to remaining work, ensuring it's non-zero
            return (num_board_needed + num_depart_needed) * 2 + 1

        # Calculate h_actions: estimate of remaining board + depart actions
        num_board_needed = len(waiting_passengers)
        # Every unserved passenger needs one 'depart' action eventually
        num_depart_needed = len(unserved_passengers)
        h_actions = num_board_needed + num_depart_needed

        # Calculate h_moves: estimate of remaining lift movement range
        h_moves = 0
        floors_to_visit = set()

        # Collect all floors the lift needs to visit for pickups and drop-offs
        # 1. Floors for picking up waiting passengers
        floors_to_visit.update(waiting_passengers.values())
        # 2. Floors for dropping off currently boarded passengers
        #    Check if passenger destination is known (should be from static facts)
        floors_to_visit.update(self.destinations[p] for p in boarded_passengers if p in self.destinations)
        # 3. Floors for eventually dropping off waiting passengers (after they board)
        floors_to_visit.update(self.destinations[p] for p in waiting_passengers if p in self.destinations)

        if floors_to_visit:
            # Include the lift's current floor when calculating the required range
            all_relevant_floors = floors_to_visit.union({lift_floor})

            # Check if floor levels were successfully computed during initialization
            # and if all currently relevant floors have a defined level.
            if not self.floor_levels or not all(f in self.floor_levels for f in all_relevant_floors):
                 # Handle cases where levels are missing (e.g., no 'above' facts, or inconsistency)
                 missing_floors = {f for f in all_relevant_floors if f not in self.floor_levels}
                 if missing_floors:
                     print(f"Warning: Floor levels undefined or missing for: {missing_floors}. Using fallback move estimate.")
                 else: # self.floor_levels was empty initially
                      print(f"Warning: Floor levels unavailable. Using fallback move estimate for floors: {all_relevant_floors}")
                 # Fallback: Estimate move cost as the number of distinct floors to visit.
                 # This ignores distance but provides a basic measure of movement complexity.
                 h_moves = len(floors_to_visit)
            else:
                # Calculate movement range using precomputed floor levels
                try:
                    relevant_levels = [self.floor_levels[f] for f in all_relevant_floors]
                    min_level = min(relevant_levels)
                    max_level = max(relevant_levels)
                    # The range is the difference between the highest and lowest needed floors
                    h_moves = max_level - min_level
                except KeyError as e:
                    # This error suggests an internal inconsistency (floor found in state/goals
                    # but level wasn't computed). Should ideally not happen with checks.
                    print(f"Internal Error: Floor level lookup failed for '{e}' despite checks. Relevant: {all_relevant_floors}. Levels: {self.floor_levels}. State: {state}")
                    # Use fallback if error occurs.
                    h_moves = len(floors_to_visit)

        # Total heuristic value is the sum of action estimate and movement estimate
        h = h_actions + h_moves
        # Ensure the heuristic value is always non-negative
        return max(0, h)

