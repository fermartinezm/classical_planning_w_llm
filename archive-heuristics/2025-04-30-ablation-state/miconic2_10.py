# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

# Helper functions (can be defined outside the class or inside if preferred,
# but outside is cleaner if they are general utilities)
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
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Assuming Heuristic base class is defined elsewhere and imported
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    The heuristic estimates the number of actions required to serve all passengers
    by summing a fixed cost for each unserved passenger based on whether they are
    waiting at their origin or already boarded. This simple heuristic implicitly
    accounts for board/depart actions and estimated movement legs per passenger.

    # Assumptions
    - Each waiting passenger requires approximately 4 actions (move to origin, board, move to destination, depart).
    - Each boarded passenger requires approximately 2 actions (move to destination, depart).
    - These costs are additive and shared movement/stops are approximated by these fixed costs per passenger state.

    # Heuristic Initialization
    - Extracts the set of all passengers who need to be served from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers who need to be served (from the goal).
    2. Identify which of these passengers are currently served (from the state).
    3. The set of unserved passengers (`U`) is the difference.
    4. If `U` is empty, the heuristic is 0 (goal state).
    5. For each passenger in `U`, determine if they are waiting at their origin floor (check for `(origin p f)` in state).
    6. Count the number of unserved passengers who are waiting (`|U_waiting|`).
    7. Count the number of unserved passengers who are boarded (`|U_boarded|`) by checking for the `(boarded p)` predicate in the state for unserved passengers.
    8. The heuristic value is calculated as `4 * |U_waiting| + 2 * |U_boarded|`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the set of passengers who need to be served.
        """
        self.goals = task.goals  # Goal conditions.
        # Identify all passengers who need to be served from the goal facts.
        self.all_goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}
        # Static facts are not explicitly needed for this simplified heuristic calculation,
        # but the base class expects it.
        self.static = task.static


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1-3. Identify unserved passengers.
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.all_goal_passengers - served_passengers
        U = unserved_passengers # Alias for clarity

        # 4. If no unserved passengers, goal is reached.
        if not U:
            return 0

        # 5-6. Count waiting passengers among the unserved.
        waiting_passengers = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1:3]
                if p in U:
                    waiting_passengers.add(p)

        U_waiting = waiting_passengers

        # 7. Count boarded passengers among the unserved.
        boarded_passengers = set()
        for fact in state:
             if match(fact, "boarded", "*"):
                 p = get_parts(fact)[1]
                 if p in U:
                     boarded_passengers.add(p)

        U_boarded = boarded_passengers

        # 8. Calculate heuristic value.
        # Each waiting passenger needs ~4 actions (move to origin, board, move to destin, depart)
        # Each boarded passenger needs ~2 actions (move to destin, depart)
        total_cost = 4 * len(U_waiting) + 2 * len(U_boarded)

        return total_cost
