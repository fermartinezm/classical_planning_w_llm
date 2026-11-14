from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    For example: "(at locatable1 locatable2)" -> ["at", "locatable1", "locatable2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (transport) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal by 
    evaluating the mismatch between the current state's facts and the goal's desired facts.
    The domain models transport along a chain of locations, where:
    - Objects called "roads" can be carried "in".
    - Locations are connected via an "at" chain.
    - The predicates (road, capacity, capacity-predecessor, in, at) change according to specific domain actions:
        * drive, pick-up, drop, drive-2.
    The goal typically requires certain objects to be "road" and at specific locations, 
    and certain "capacity-predecessor" conditions to hold.

    Assumptions
    -----------
    - The provided state and goal consist of facts with predicates "road", "at", and "capacity-predecessor".
    - The "capacity" and "in" predicates represent a capacity token carried by one object at a time.
    - The heuristic does not guarantee admissibility but strives to estimate distance to goal efficiently.
    - The chain structure modeled by "at" predicates behaves like linked locations; objects move along it.
    - The goal typically fixes a sequence of "road" and "at" relations between objects.
    - Static facts are empty in this domain, so no static info is used.

    Heuristic Initialization
    ------------------------
    - Extract the goal facts.
    - Identify from the goal the sets of:
        * Objects that must have (road ?obj) true.
        * Objects that must have (capacity-predecessor ?obj) true.
        * Pairs of objects that must satisfy (at ?obj1 ?obj2).
    - These sets serve as references when comparing the current state facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the state is a goal state (all goal facts hold), return 0.

    2. Count how many goal-required "road" facts are missing in the current state.
       Each missing "road" fact likely requires at least one action to establish.

    3. Count how many goal-required "capacity-predecessor" facts are missing in the current state.
       Each missing capacity-predecessor fact indicates the capacity token needs to be moved or adjusted.

    4. Count how many goal-required "at" facts are missing in the current state.
       Since "at" predicates form a chain linking objects, each missing link is an action needed to move objects.

    5. Consider the presence of "(in ?obj)" predicate in the current state (an object carrying capacity).
       If capacity is currently "in" an object that is not matching the goal capacity location, add 1 to heuristic
       to account for the action required to move or drop capacity.

    6. Sum all these counts into a heuristic value estimating the minimum number of actions needed.

    7. The heuristic favors states closer to having correct "road", "capacity-predecessor", and "at" facts, 
       encouraging expansion towards goal chains.

    This heuristic attempts a simple distance measure over the key predicates controlling the state, 
    balancing accuracy with computational efficiency.
    """

    def __init__(self, task):
        # Extract goal facts grouped by predicate and relevant objects
        self.goal_road = set()
        self.goal_capacity_pred = set()
        self.goal_at_pairs = set()

        for fact in task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "road" and len(parts) == 2:
                self.goal_road.add(parts[1])
            elif pred == "capacity-predecessor" and len(parts) == 2:
                self.goal_capacity_pred.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                self.goal_at_pairs.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero
        if all(
            (
                f if not f.startswith("(at ") else f in state
            )
            for f in node.task.goals
        ):
            # Defensive: exact goal check using subset condition:
            if node.task.goals <= state:
                return 0

        # Extract current facts by predicate for fast lookup
        current_road = set()
        current_capacity_pred = set()
        current_at_pairs = set()
        in_objects = set()  # Objects currently carrying capacity (have (in ?obj))

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "road" and len(parts) == 2:
                current_road.add(parts[1])
            elif pred == "capacity-predecessor" and len(parts) == 2:
                current_capacity_pred.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                current_at_pairs.add((parts[1], parts[2]))
            elif pred == "in" and len(parts) == 2:
                in_objects.add(parts[1])

        # Count missing goal facts by predicate
        missing_road = len(self.goal_road - current_road)
        missing_capacity_pred = len(self.goal_capacity_pred - current_capacity_pred)
        missing_at = len(self.goal_at_pairs - current_at_pairs)

        # Check if capacity is currently held by any object NOT in goal capacity-predecessor locations,
        # indicating the capacity token must still be moved
        capacity_in_wrong_place = 0
        if in_objects:
            # The capacity token is carried by one or more objects (ideally exactly one)
            # If the carrier object is not in goal_capacity_pred, then we likely need an action to move capacity
            if not (in_objects & self.goal_capacity_pred):
                capacity_in_wrong_place = 1

        # Calculate heuristic value as sum of missing essential facts + possible cost of moving capacity token
        heuristic_value = missing_road + missing_capacity_pred + missing_at + capacity_in_wrong_place

        return heuristic_value
