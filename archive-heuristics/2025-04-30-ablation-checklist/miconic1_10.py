from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully, though PDDL states are structured.
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces.
    # Use shlex.split for more robust parsing if facts could contain spaces/quotes,
    # but simple split is usually sufficient for standard PDDL fact strings like '(pred obj1 obj2)'.
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort by summing:
    1. The number of 'board' actions needed for passengers waiting at their origin.
    2. The number of 'depart' actions needed for passengers currently boarded.
    3. The number of distinct floors that must be visited to pick up waiting passengers
       or drop off boarded passengers.

    # Assumptions
    - All passengers listed in the goal must be served.
    - Passengers are either waiting at their origin, boarded, or served.
    - The cost of moving between floors is implicitly captured by the number of distinct floors requiring a stop.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Identifies all passengers that need to be served based on the goal state.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Parse the current state to quickly access facts like `(served ?p)`, `(origin ?p ?f)`, and `(boarded ?p)`.
    2. Initialize the heuristic value `h = 0`.
    3. Initialize an empty set `required_floors` to track floors that need a stop (for pickup or dropoff).
    4. Iterate through each passenger that needs to be served (identified during initialization).
    5. For an unserved passenger `p`:
       - If `p` is currently waiting at their origin floor `f` (`(origin p f)` is true):
         - Increment `h` by 1 (for the `board` action).
         - Add floor `f` to the `required_floors` set.
       - If `p` is currently boarded (`(boarded p)` is true):
         - Increment `h` by 1 (for the `depart` action).
         - Find the destination floor `d` for `p` (using the precomputed destination map).
         - Add floor `d` to the `required_floors` set.
    6. Add the number of distinct floors in the `required_floors` set to `h`. This term estimates the movement cost.
    7. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and
        identifying passengers that need to be served.
        """
        self.goals = task.goals  # Goal conditions.
        self.static_facts = task.static # Static facts.

        # Map passengers to their destination floors from static facts.
        self.destinations = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin':
                # Fact is like '(destin passenger floor)'
                if len(parts) == 3:
                    self.destinations[parts[1]] = parts[2]
                # else: Handle unexpected format? Assume valid PDDL.

        # Identify the set of passengers that must be served to reach the goal.
        self.passengers_to_serve = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'served':
                 # Goal fact is like '(served passenger)'
                 if len(parts) == 2:
                    self.passengers_to_serve.add(parts[1])
                 # else: Handle unexpected format? Assume valid PDDL.


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic estimate for the given state.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Parse state facts for easier lookup
        state_predicates = {}
        # Store facts by predicate name, handling multiple instances
        for fact in state:
            parts = get_parts(fact)
            if parts:
                pred = parts[0]
                if pred not in state_predicates:
                    state_predicates[pred] = []
                state_predicates[pred].append(parts)

        # Get sets of served, origin, and boarded passengers/locations
        served_passengers = {p for [pred, p] in state_predicates.get('served', [])}
        origin_locations = {p: f for [pred, p, f] in state_predicates.get('origin', [])}
        boarded_passengers = {p for [pred, p] in state_predicates.get('boarded', [])}
        # Note: lift_at is not strictly needed for this heuristic calculation,
        # but could be extracted: lift_location = state_predicates.get('lift-at', [[None, None]])[0][1]


        h = 0  # Initialize heuristic value
        required_floors = set() # Floors that need a stop

        # Iterate through passengers who need to reach their destination
        for passenger in self.passengers_to_serve:
            # If the passenger is already served, no more actions needed for them
            if passenger in served_passengers:
                continue

            # If the passenger is waiting at their origin
            if passenger in origin_locations:
                # Needs a 'board' action
                h += 1
                origin_floor = origin_locations[passenger]
                required_floors.add(origin_floor)

            # If the passenger is currently boarded
            elif passenger in boarded_passengers:
                 # Needs a 'depart' action
                 h += 1
                 # Get their destination floor (precomputed)
                 destination_floor = self.destinations.get(passenger)
                 if destination_floor: # Should always exist for goal passengers
                     required_floors.add(destination_floor)
                 # else: Handle error? Assume valid problem definition.

            # Note: A passenger should be either at origin, boarded, or served.
            # If not served and not at origin, they must be boarded (assuming valid states).
            # The logic above covers these cases.

        # Add the number of distinct floors that need to be visited.
        # This serves as a proxy for the movement cost.
        h += len(required_floors)

        return h

