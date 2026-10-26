from fnmatch import fnmatch
# Assuming heuristic_base.py is available and defines a Heuristic base class
# from heuristics.heuristic_base import Heuristic
import re # For natural sorting

# Define a dummy Heuristic base class if not provided in the execution environment
# In a real scenario, this would be imported from the planner's framework.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

# Utility functions (copied and adapted from examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def natural_sort_key(s):
    """
    Key function for natural sorting of strings containing numbers.
    Useful for sorting floor names like f1, f2, f10 correctly.
    """
    # Split the string into parts of digits and non-digits
    parts = [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    # Filter out any empty strings that might result from the split
    return [part for part in parts if part != '']


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    This heuristic estimates the total number of actions required to reach the
    goal state (all passengers served) by summing the estimated cost for each
    unserved passenger independently. This is a non-admissible heuristic suitable
    for greedy best-first search.

    The estimated cost for a single unserved passenger is calculated as follows:
    - If the passenger is waiting at their origin floor O:
      Cost = (moves from current lift floor L to O) + board + (moves from O to destination D) + depart
      = |L_int - O_int| + 1 + |O_int - D_int| + 1
    - If the passenger is already boarded (and not served), they must be in the lift:
      Cost = (moves from current lift floor L to destination D) + depart
      = |L_int - D_int| + 1

    Floor movements are estimated by the absolute difference in floor indices.
    Floor indices are determined by parsing all floor names mentioned in the
    problem and sorting them naturally (e.g., f1, f2, f10). This assumes floor
    names correspond to a linear order where natural sort matches the 'above'
    relation order.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and
        creating a mapping from floor names to integer indices based on sorting.

        Args:
            task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Parse passenger destinations from static facts.
        # The 'destin' predicate is static.
        self.destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure fact has correct number of parts
                    _, passenger, floor = parts
                    self.destinations[passenger] = floor
                # else: Log a warning about malformed destin fact?

        # 2. Parse floor order and create floor_to_int map.
        # We collect all floor names mentioned in the static facts and initial state,
        # and sort them naturally to establish an order and assign integer indices.
        floor_names = set()
        # Collect floors from static facts (above, destin)
        for fact in self.static:
             parts = get_parts(fact)
             # Consider predicates that typically involve floors
             if parts and parts[0] in ["above", "destin"]:
                  for part in parts[1:]:
                       # Simple check if the part looks like a floor name (starts with 'f', has more chars)
                       if isinstance(part, str) and part.startswith('f') and len(part) > 1:
                            floor_names.add(part)
        # Collect floors from initial state (lift-at, origin)
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Consider predicates that typically involve floors
             if parts and parts[0] in ["lift-at", "origin"]:
                  for part in parts[1:]:
                       # Simple check if the part looks like a floor name
                       if isinstance(part, str) and part.startswith('f') and len(part) > 1:
                            floor_names.add(part)

        # Sort floors naturally (e.g., f1, f2, ..., f10, f11, ...)
        # This assumes floor names are consistently formatted and natural sort
        # aligns with the 'above' relation defined in the domain.
        sorted_floor_names = sorted(list(floor_names), key=natural_sort_key)

        self.floor_to_int = {f: i for i, f in enumerate(sorted_floor_names)}
        # self.int_to_floor = sorted_floor_names # Optional: store for debugging

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining actions, or float('inf')
            if the state is deemed invalid or unreachable.
        """
        state = node.state

        # If the goal is already reached, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Find the current floor of the lift.
        current_lift_floor_name = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2: # Ensure fact has correct number of parts
                    current_lift_floor_name = parts[1]
                    break

        # If lift location is unknown or not a recognized floor, return infinity.
        # This indicates an unexpected or potentially invalid state.
        if current_lift_floor_name is None or current_lift_floor_name not in self.floor_to_int:
             return float('inf')

        current_lift_floor_int = self.floor_to_int[current_lift_floor_name]

        h = 0
        # Identify passengers who are already served to exclude them.
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Iterate through all passengers whose destination is known from static facts.
        # These are the only passengers relevant to the goal.
        for passenger, dest_name in self.destinations.items():
            # If the passenger is already served, they don't contribute to the heuristic cost.
            if passenger in served_passengers:
                continue

            # Get the integer index for the destination floor.
            dest_int = self.floor_to_int.get(dest_name)
            # If destination floor is not recognized, something is wrong with the problem definition.
            # This check is mostly defensive after parsing floors in __init__.
            if dest_int is None:
                 return float('inf')

            # Determine the passenger's current status: waiting at origin or boarded.
            # A passenger is either served, waiting at origin, or boarded.
            # If not served, we check if they are waiting at their origin.
            # If not served and not waiting, they must be boarded.
            is_waiting = False
            origin_name = None
            # Iterate through state facts to find if the passenger is at their origin.
            # We assume origin facts are like (origin passenger floor).
            for fact in state:
                 if match(fact, "origin", passenger, "*"):
                      parts = get_parts(fact)
                      if len(parts) == 3: # Ensure fact has correct number of parts
                           is_waiting = True
                           origin_name = parts[2]
                           break
                 # No need to explicitly check for (boarded p) here, as it's the implicit alternative
                 # if the passenger is not served and not found at an origin.

            if is_waiting:
                # Passenger is waiting at their origin floor.
                origin_int = self.floor_to_int.get(origin_name)
                # If origin floor is not recognized, something is wrong.
                if origin_int is None:
                     return float('inf')

                # Estimated cost for a waiting passenger:
                # 1. Move from current lift floor to origin: abs(current_lift_floor_int - origin_int) moves
                # 2. Board the passenger: +1 action
                # 3. Move from origin to destination: abs(origin_int - dest_int) moves
                # 4. Depart the passenger: +1 action
                h += abs(current_lift_floor_int - origin_int) + 1 + abs(origin_int - dest_int) + 1
            else:
                # Passenger is not served and not waiting at origin, so they must be boarded.
                # Estimated cost for a boarded passenger:
                # 1. Move from current lift floor to destination: abs(current_lift_floor_int - dest_int) moves
                # 2. Depart the passenger: +1 action
                h += abs(current_lift_floor_int - dest_int) + 1

        return h

