# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the specific environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

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
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions for unserved passengers
    and adds an estimate of the travel cost for the lift to visit all necessary floors.

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < ...).
    - The 'above' facts define this linear order, and up/down actions move between
      immediately adjacent floors in this order (although the PDDL doesn't strictly
      enforce immediate adjacency for the precondition, the standard interpretation
      and typical instances imply it for movement actions).
    - The lift can carry multiple passengers.
    - Unserved passengers are either waiting at their origin or are boarded.

    # Heuristic Initialization
    - Parses static facts to determine the floor ordering and create a mapping
      from floor names to numerical levels. Assumes floor names are in 'f<number>' format.
    - Parses static and initial state facts to identify all passengers and their
      destination floors.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all passengers and their destination floors from the task definition
       (using static facts).
    2. Map floor names to numerical levels based on the 'above' static facts.
       Assumes floor names are parseable as 'f' followed by a number.
    3. In the heuristic function (`__call__`):
       a. Check if the goal is reached (all passengers served). This is the state
          where the heuristic should be 0.
       b. Identify all passengers who are not yet served.
       c. Initialize the heuristic value `h` to 0.
       d. Initialize a set `required_levels` to store the numerical levels of floors
          the lift must visit.
       e. Find the lift's current floor from the state and get its numerical level.
       f. Iterate through the unserved passengers:
          - Add 1 to `h` for the eventual `depart` action this passenger needs.
          - Determine the passenger's current status (waiting at origin or boarded)
            by checking the state facts.
          - If the passenger is currently waiting at their origin floor:
            - Add 1 to `h` for the eventual `board` action this passenger needs.
            - Add the level of the passenger's origin floor to `required_levels`.
          - If the passenger is currently boarded:
            - Retrieve the passenger's destination floor (from initialization data).
            - Add the level of the passenger's destination floor to `required_levels`.
       g. If `required_levels` is not empty:
          - Find the minimum and maximum levels among the `required_levels`.
          - Estimate the travel cost as the difference between the maximum of
            (current lift level and max required level) and the minimum of
            (current lift level and min required level). This represents the
            total vertical span the lift must cover to potentially visit all
            required floors starting from its current position.
          - Add this estimated travel cost to `h`.
       h. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.task = task
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Identify all passengers and their destination floors
        self.passenger_destin = {}
        all_passengers = set()
        all_floors = set()

        # Get passenger destinations from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                _, p, f = parts
                self.passenger_destin[p] = f
                all_passengers.add(p)

        # Also collect passengers and floors from initial state origin facts
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "origin":
                 _, p, f = parts
                 all_passengers.add(p)
                 all_floors.add(f) # Collect floors here too

        # 2. Map floor names to numerical levels
        # Collect all floors mentioned in static 'above' facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                _, f1, f2 = parts
                all_floors.add(f1)
                all_floors.add(f2)

        # Assuming floor names are like f1, f2, ... and can be sorted numerically
        # This is a domain-specific assumption based on typical benchmarks.
        try:
            # Sort floors based on the number part of the name
            sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
            # Create a mapping from floor name to its numerical level (1-based index)
            self.floor_to_level = {f: i + 1 for i, f in enumerate(sorted_floors)}
        except (ValueError, IndexError):
            # Fallback if floor names are not in the expected 'f<number>' format
            # This fallback assigns an arbitrary order and might lead to a poor heuristic.
            print("Warning: Floor names not in f<number> format or parsing failed. Using arbitrary floor order.")
            self.floor_to_level = {f: i for i, f in enumerate(sorted(list(all_floors)))} # Use alphabetical sort as fallback


        self.all_passengers = list(all_passengers)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 3a. Check if the goal is reached
        # The goal is that all passengers are served.
        all_served = True
        for p in self.all_passengers:
            if '(served ' + p + ')' not in state:
                all_served = False
                break
        if all_served:
            return 0

        # 3b. Identify unserved passengers
        unserved_passengers = {p for p in self.all_passengers if '(served ' + p + ')' not in state}

        # 3c. Initialize heuristic value
        h = 0

        # 3d. Initialize required floors/levels
        required_levels = set()
        lift_current_floor = None

        # Find lift's current floor and identify passenger states
        waiting_passengers_at_origin = {} # Map passenger to origin floor if waiting
        boarded_passengers = set() # Set of boarded passengers

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "lift-at":
                lift_current_floor = parts[1]
            elif predicate == "origin":
                p, f_o = parts[1], parts[2]
                if p in unserved_passengers:
                    waiting_passengers_at_origin[p] = f_o
            elif predicate == "boarded":
                 p = parts[1]
                 if p in unserved_passengers:
                     boarded_passengers.add(p)

        # If for some reason lift_current_floor wasn't found (invalid state?),
        # the heuristic calculation involving travel won't be possible.
        # Assuming valid states always have lift-at.
        if lift_current_floor is None:
             # This indicates an unexpected state. Return a high value or handle as error.
             # For a heuristic, returning a large number might guide search away.
             # However, if there are unserved passengers, lift *must* be somewhere.
             # Let's assume lift_current_floor is always found if unserved > 0.
             # If unserved == 0, we already returned 0.
             # If lift_current_floor is None and unserved > 0, this is an invalid state.
             # Returning a large number might be appropriate, but let's assume valid states.
             pass # Should not happen in valid reachable states with unserved passengers

        # 3f. Iterate through unserved passengers and determine required stops and actions
        for p in unserved_passengers:
             # Each unserved passenger needs a depart action
             h += 1

             if p in waiting_passengers_at_origin:
                 # Needs a board action
                 h += 1
                 f_o = waiting_passengers_at_origin[p]
                 if f_o in self.floor_to_level:
                     required_levels.add(self.floor_to_level[f_o])
                 # else: Floor not mapped? Should not happen if init parsing is correct.

             elif p in boarded_passengers:
                 # Needs to be dropped off at destination
                 if p in self.passenger_destin:
                     f_d = self.passenger_destin[p]
                     if f_d in self.floor_to_level:
                         required_levels.add(self.floor_to_level[f_d])
                     # else: Floor not mapped? Should not happen.
                 # else: Passenger destination not found? Should not happen if init parsing is correct.
             # else: Unserved passenger is neither waiting nor boarded. This is an invalid state
             # in standard miconic. We assume valid states.


        # 3g. Calculate travel cost
        if required_levels:
            # Ensure lift_current_floor was found and is a known floor
            current_level = self.floor_to_level.get(lift_current_floor)
            if current_level is not None:
                min_req = min(required_levels)
                max_req = max(required_levels)

                # Estimate travel as the total vertical span including the current floor
                # This is max(current_level, max_req) - min(current_level, min_req)
                # It represents the minimum range of floors the lift must traverse.
                travel_cost_estimate = max(current_level, max_req) - min(current_level, min_req)
                h += travel_cost_estimate
            # else: Cannot calculate travel if lift location is unknown or unmapped.
            # The heuristic will just be the sum of board/depart actions in this case.


        # 3h. Return total heuristic value
        return h
