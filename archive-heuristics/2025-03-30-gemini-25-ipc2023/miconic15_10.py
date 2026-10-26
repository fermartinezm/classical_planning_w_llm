import math
from fnmatch import fnmatch
# The planner infrastructure should provide the Heuristic base class.
# If running standalone, you might need to define a placeholder, e.g.:
# class Heuristic:
#     def __init__(self, task): self.task = task
#     def __call__(self, node): raise NotImplementedError
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str) -> list:
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes surrounding parentheses and splits by space.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    # Ensure fact is a non-empty string starting with '(' and ending with ')'
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return an empty list or raise an error for invalid format
        # print(f"Warning: Invalid fact format encountered: {fact}")
        return []
    return fact[1:-1].split()

def match(fact_parts: list, *pattern: str) -> bool:
    """
    Checks if the parts of a fact (already split) match a given pattern.
    Allows '*' wildcard matching using fnmatch for each part.
    Returns False if the number of parts does not match the pattern length.
    """
    # Check if the number of parts matches the pattern length
    if len(fact_parts) != len(pattern):
        return False
    # Check each part against the corresponding pattern element using fnmatch
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    specified in the goal. It calculates an estimated cost for each unserved
    passenger individually and sums these costs. The cost for a passenger includes
    the estimated lift movement needed to pick them up (if they are waiting at their
    origin) and drop them off at their destination, plus the required board and
    depart actions.

    # Assumptions
    - The `(above f1 f2)` predicate means floor `f1` is at a higher level than
      floor `f2` (transitively, not necessarily directly above). Example: If F3 is above F2 and F2 is above F1, then (above F3 F2), (above F2 F1), and (above F3 F1) might all be true in the static facts.
    - Floor levels can be derived numerically based on the count of floors below them
      in the 'above' relation hierarchy. The cost of moving between floors is the
      absolute difference in their levels (1 action per level difference).
    - The heuristic sums the estimated costs for each passenger independently.
      This does not account for potential synergies (multiple passengers sharing
      a lift ride) and thus is likely non-admissible, but aims to provide a
      useful estimate for guiding greedy search.
    - All passengers mentioned in the goal must eventually be in the 'served' state.

    # Heuristic Initialization
    - Parses static facts (`task.static`) upon initialization.
    - Determines the numerical level of each floor based on the `(above f1 f2)`
      relationships found in the static facts. The level of a floor `f` is calculated
      as the count of other floors `f'` for which `(above f f')` is true.
      The lowest floor(s) (those with no floors below them) will have level 0.
      This mapping from floor name to level is stored in `self.floor_levels`.
    - Stores the destination floor for each passenger using the static `(destin p f)`
      predicates in `self.destinations`.
    - Identifies the set of passengers that need to be served according to the
      goal conditions (`task.goals`) and stores them in `self.goal_passengers`.
    - Dynamically adds floors encountered during initialization (e.g., from destin)
      to the level map if they weren't mentioned in 'above' relations, assigning them level 0.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state (`node.state`).
    2. Find the current floor of the lift (`f_lift`) by looking for the `(lift-at ?f)` fact in the state. Handle cases where the lift's floor or its level might be unknown (assign level 0 dynamically if needed).
    3. Identify the status of all passengers in the current state:
       - Which passengers `p` are waiting at their origin `f_orig` (`(origin p f_orig)` facts)? Store in `waiting_passengers` dict (p -> f_orig). Handle potentially unknown origin floor levels.
       - Which passengers `p` are currently inside the lift (`(boarded p)` facts)? Store in `boarded_passengers` set.
       - Which passengers `p` have already reached their destination (`(served p)` facts)? Store in `served_passengers` set.
    4. Determine the set of passengers who are required by the goal but are not yet served (`unserved_goal_passengers = self.goal_passengers - served_passengers`).
    5. If `unserved_goal_passengers` is empty, the goal state is reached, return 0.
    6. Initialize the total heuristic cost `h = 0`.
    7. For each passenger `p` in `unserved_goal_passengers`:
       a. Get their destination floor `f_dest` from the precomputed `self.destinations`. Handle cases where the destination or its level is unknown.
       b. **If `p` is currently waiting at `f_orig` (i.e., `p` is in `waiting_passengers`):**
          - Estimate cost as:
            - Lift movement from `f_lift` to `f_orig`: `dist(f_lift, f_orig)`
            - Board action: `+ 1`
            - Lift movement from `f_orig` to `f_dest`: `dist(f_orig, f_dest)`
            - Depart action: `+ 1`
          - Add this total (`dist(f_lift, f_orig) + dist(f_orig, f_dest) + 2`) to `h`.
       c. **If `p` is currently boarded (i.e., `p` is in `boarded_passengers`):**
          - Estimate cost as:
            - Lift movement from `f_lift` to `f_dest`: `dist(f_lift, f_dest)`
            - Depart action: `+ 1`
          - Add this total (`dist(f_lift, f_dest) + 1`) to `h`.
       d. **If `p` is unserved but neither waiting nor boarded:** This indicates an unexpected or potentially invalid state. Add a fixed penalty (e.g., 4) to `h` and print a warning message.
    8. The distance `dist(f1, f2)` is calculated using the precomputed floor levels: `abs(self.floor_levels[f1] - self.floor_levels[f2])`. The helper method `_get_floor_level` handles potentially unknown floors encountered during runtime by assigning them level 0.
    9. Return the final sum `h`, ensuring it's non-negative.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.floor_levels = {}
        self.destinations = {}
        self.goal_passengers = set()

        all_floors = set()
        above_relations = []

        # --- Pass 1: Gather all floors, destinations, and 'above' relations ---
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip if fact format was invalid

            if match(parts, "above", "*", "*"):
                f1, f2 = parts[1], parts[2]
                all_floors.add(f1)
                all_floors.add(f2)
                above_relations.append((f1, f2))
            elif match(parts, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
                all_floors.add(floor) # Ensure destination floor is tracked

        # --- Pass 2: Calculate floor levels ---
        # Initialize all tracked floors with level 0
        for f in all_floors:
            self.floor_levels[f] = 0
        # Update levels based on 'above' relations count
        for f in all_floors:
            level = 0
            # Level is the count of distinct floors strictly below f
            floors_below = set()
            for f1_rel, f2_rel in above_relations:
                if f1_rel == f:
                    floors_below.add(f2_rel)
            self.floor_levels[f] = len(floors_below)

        # --- Identify Goal Passengers ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid goal format

            if match(parts, "served", "*"):
                self.goal_passengers.add(parts[1])

        # Debug: Print computed levels and destinations
        # print("Computed Floor Levels:", self.floor_levels)
        # print("Passenger Destinations:", self.destinations)
        # print("Goal Passengers:", self.goal_passengers)


    def _get_floor_level(self, floor: str) -> int:
        """
        Safely retrieves the level of a floor.
        If the floor was not seen during initialization (e.g., only appears
        in the initial state's lift-at or origin), it assigns level 0 dynamically
        and prints a warning.
        """
        if floor not in self.floor_levels:
            # This might happen if a floor appears only in the initial state
            # (e.g., lift-at, origin) and not in any static 'above' or 'destin'.
            print(f"Warning: Floor level was not precomputed for floor '{floor}'. Assigning level 0.")
            self.floor_levels[floor] = 0 # Assign default level dynamically
        return self.floor_levels[floor]

    def _get_floor_distance(self, f1: str, f2: str) -> int:
        """
        Calculates the distance (number of moves) between two floors using their levels.
        Uses _get_floor_level to handle potentially unknown floors.
        """
        level1 = self._get_floor_level(f1)
        level2 = self._get_floor_level(f2)
        return abs(level1 - level2)

    def __call__(self, node) -> int:
        """
        Calculates the heuristic value for the given state node.
        Returns an integer estimate of the remaining actions.
        """
        state = node.state

        # --- Find current lift location ---
        lift_at_floor = None
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if match(parts, "lift-at", "*"):
                lift_at_floor = parts[1]
                break # Found the lift location

        if lift_at_floor is None:
            # This should not happen in a valid Miconic problem state.
            print("Error: Lift location predicate (lift-at) not found in state. Cannot compute heuristic.")
            # Return infinity to indicate an invalid or error state for search.
            return float('inf')

        # --- Identify passenger states from the current state ---
        waiting_passengers = {} # passenger -> origin_floor
        boarded_passengers = set()
        served_passengers = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if match(parts, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                waiting_passengers[passenger] = floor
            elif match(parts, "boarded", "*"):
                passenger = parts[1]
                boarded_passengers.add(passenger)
            elif match(parts, "served", "*"):
                passenger = parts[1]
                served_passengers.add(passenger)

        # --- Calculate heuristic value ---
        heuristic_value = 0
        # Find passengers required by the goal who are not yet served
        unserved_goal_passengers = self.goal_passengers - served_passengers

        # If the set is empty, all goal passengers are served.
        if not unserved_goal_passengers:
            return 0

        # Calculate cost for each unserved goal passenger
        for p in unserved_goal_passengers:
            # Ensure passenger has a known destination from static facts
            if p not in self.destinations:
                 print(f"Warning: Destination unknown for goal passenger '{p}'. Skipping cost calculation for this passenger.")
                 continue # Skip this passenger if their destination isn't defined

            dest_floor = self.destinations[p]

            if p in waiting_passengers:
                # Passenger is waiting at their origin floor
                origin_floor = waiting_passengers[p]
                # Cost = move(lift -> origin) + board + move(origin -> dest) + depart
                cost_to_origin = self._get_floor_distance(lift_at_floor, origin_floor)
                cost_origin_to_dest = self._get_floor_distance(origin_floor, dest_floor)
                heuristic_value += cost_to_origin + 1 + cost_origin_to_dest + 1
            elif p in boarded_passengers:
                # Passenger is already inside the lift
                # Cost = move(lift -> dest) + depart
                cost_to_dest = self._get_floor_distance(lift_at_floor, dest_floor)
                heuristic_value += cost_to_dest + 1
            else:
                # Passenger 'p' is required for the goal and not served, but is
                # neither waiting at origin nor boarded. This indicates an unexpected state.
                print(f"Warning: Unserved goal passenger '{p}' is neither waiting nor boarded. Applying fixed penalty.")
                # Add a fixed penalty, assuming something went wrong or state is unusual.
                # This might happen if the problem definition allows states not reachable
                # through standard action sequences, or if there's an issue elsewhere.
                heuristic_value += 4 # Arbitrary penalty (e.g., board + depart + 2 moves)

        # Return the calculated heuristic value, ensuring it's non-negative.
        # The sum naturally should be non-negative if distances and action counts are.
        return heuristic_value
