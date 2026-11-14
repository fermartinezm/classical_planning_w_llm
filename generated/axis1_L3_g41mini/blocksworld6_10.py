from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain 'transport' domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state
    by analyzing the distribution of key predicates in the current state versus the goal.
    It leverages the relationships between road, capacity-predecessor, capacity, in, and at
    predicates, which encode capacity and position information. It estimates how many
    "drive", "pick-up", "drop", and "drive-2" actions will be necessary to satisfy
    the goal conditions.

    Assumptions:
    - The heuristic assumes that goal conditions consist primarily of facts over predicates
      road, capacity-predecessor, and at.
    - Each "in" fact indicates an object currently being transported.
    - "capacity-predecessor" facts indicate objects involved in a chain linked to capacity.
    - Actions like "drive" and "drive-2" require specific conditions on road, capacity-predecessor, and capacity.
    - Heuristic need not be admissible but must be informative and efficient.
    - Static facts are empty in this domain.

    Heuristic Initialization:
    - Extract the goal sets of facts over predicates: road, capacity-predecessor, and at.
    - No static facts are relevant.
    - Store goal sets for quick membership testing.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state is a goal state, return 0 immediately.
    2. Count how many objects in the goal have the capacity-predecessor predicate but are currently missing it.
       Each missing capacity-predecessor likely requires acting with "pick-up" or "drop".
    3. Count how many road predicates present in the goal are missing in the current state.
       This corresponds to objects needing "road" added or restored via "drop" or "pick-up".
    4. Count how many "at" goal facts are not currently achieved.
       Each misplaced "at" fact means the object will need moving via "drive-2" or "drop".
    5. Count how many objects currently are "in" (being carried).
       These probably must be dropped.
    6. Because the domain allows two kinds of drive actions (drive and drive-2), and pick-up/drop
       modify capacity predicates, estimate needed actions as the sum of missing capacity-predecessor,
       missing road, misplaced at, and objects currently in transport.
    7. Assign heuristic cost with these counts weighted roughly by 1, since each missing predicate
       roughly means one or a few actions. Adding counts provides a cost estimate guiding search.
    8. Return the computed heuristic estimate (non-zero for non-goal states).

    The heuristic ensures a finite number by always summing counts derived from current and goal predicates.
    """

    def __init__(self, task):
        # Store the goal sets by predicate for quick membership testing.
        self.goal_road = set()
        self.goal_capacity_pred = set()
        self.goal_at = set()

        for fact in task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "road":
                self.goal_road.add(parts[1])
            elif pred == "capacity-predecessor":
                self.goal_capacity_pred.add(parts[1])
            elif pred == "at":
                # at predicate has two arguments: at ?v1 ?v2
                if len(parts) == 3:
                    self.goal_at.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # If this is a goal state, heuristic is zero
        # goals are sets of facts as strings, state is frozenset of facts
        if all(
            (fact in state)
            for fact in node.task.goals
        ):
            return 0

        # Parse current state for predicates relevant in the heuristic
        current_road = set()
        current_capacity_pred = set()
        current_at = set()
        current_in = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "road" and len(parts) == 2:
                current_road.add(parts[1])
            elif pred == "capacity-predecessor" and len(parts) == 2:
                current_capacity_pred.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                current_at.add((parts[1], parts[2]))
            elif pred == "in" and len(parts) == 2:
                current_in.add(parts[1])

        # Count missing capacity-predecessor objects in goal vs current state
        missing_capacity_pred = len(self.goal_capacity_pred - current_capacity_pred)

        # Count missing road objects in goal vs current state
        missing_road = len(self.goal_road - current_road)

        # Count misplaced 'at' relationships: goal (obj, loc) pairs missing in current state
        missing_at = len(self.goal_at - current_at)

        # Count objects currently 'in' transport (to be dropped eventually)
        in_count = len(current_in)

        # Estimate heuristic cost as sum of these counts:
        # Each missing capacity-predecessor or road likely needs pick-up or drop actions.
        # Each missing 'at' likely requires one or more move/drop actions.
        # Each 'in' means an object currently carried that must be dropped.
        heuristic_value = missing_capacity_pred + missing_road + missing_at + in_count

        return heuristic_value
