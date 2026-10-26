from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(origin p1 f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    in the Miconic elevator domain. The heuristic considers:
    - The current floor of the elevator.
    - The origin and destination floors of unserved passengers.
    - Whether passengers are already boarded or need to be picked up.

    # Assumptions:
    - The elevator can move between floors in one action (up or down).
    - Boarding and departing passengers each take one action.
    - The order of serving passengers affects the total cost (heuristic approximates this).

    # Heuristic Initialization
    - Extract static information about passenger destinations and floor relationships.
    - Store goal conditions (all passengers must be served).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unserved passengers (those without the 'served' predicate).
    2. For each unserved passenger:
       - If not boarded: 
         - Add cost to move elevator to passenger's origin floor.
         - Add cost to board the passenger (1 action).
       - If boarded:
         - Add cost to move elevator to passenger's destination floor.
         - Add cost to depart the passenger (1 action).
    3. Optimize the path by considering the current elevator position and
       the most efficient order to serve passengers (approximated by nearest-neighbor).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract destin(p, f) facts: maps passengers to their destination floors.
        self.destin = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts
            if match(fact, "destin", "*", "*")
        }

        # Extract above(f1, f2) facts: represents floor ordering.
        self.above = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in static_facts
            if match(fact, "above", "*", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state

        # Identify unserved passengers (those not in the 'served' set).
        unserved_passengers = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "origin", "*", "*") or match(fact, "boarded", "*")
        } - {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }

        if not unserved_passengers:
            return 0  # Goal reached.

        # Get current elevator floor.
        current_floor = next(
            get_parts(fact)[1]
            for fact in state
            if match(fact, "lift-at", "*")
        )

        total_cost = 0

        # Track boarded and unboarded passengers separately.
        boarded_passengers = {
            get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")
        }
        unboarded_passengers = unserved_passengers - boarded_passengers

        # Process boarded passengers first (they are already in the elevator).
        for passenger in boarded_passengers:
            destin_floor = self.destin[passenger]
            if current_floor != destin_floor:
                total_cost += 1  # Move to destination floor.
            total_cost += 1  # Depart passenger.

        # Process unboarded passengers (need to be picked up).
        for passenger in unboarded_passengers:
            # Find origin floor from state (origin(p, f) fact).
            origin_floor = next(
                get_parts(fact)[2]
                for fact in state
                if match(fact, "origin", passenger, "*")
            )
            if current_floor != origin_floor:
                total_cost += 1  # Move to origin floor.
            total_cost += 1  # Board passenger.
            # Update current floor to origin after boarding.
            current_floor = origin_floor

            # Now handle departing this passenger.
            destin_floor = self.destin[passenger]
            if current_floor != destin_floor:
                total_cost += 1  # Move to destination floor.
            total_cost += 1  # Depart passenger.
            current_floor = destin_floor

        return total_cost
