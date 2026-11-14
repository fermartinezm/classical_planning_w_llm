from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (transport) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to transform the current state
    into a goal state. The domain models a transportation-like problem where each "locatable"
    either has the predicates (road ?loc) or (capacity-predecessor ?loc), and the key predicates
    are (road), (capacity-predecessor), (capacity), (in), and (at ?loc1 ?loc2).
    
    The actions cycle around driving along roads, picking up and dropping capacity along locations.
    The heuristic primarily focuses on matching the predicates required by the goals, and estimating
    how many actions are needed to fix discrepancies between the current state and the goal state.
    
    Assumptions
    -----------
    - Each locatable can have only certain predicates at once, and the actions manipulate those predicates.
    - The goal requires some locatables to have (road ?v) and some to be at others (at ?v1 ?v2),
      and some locatables to have (capacity-predecessor ?v).
    - The "capacity" and "in" predicate are transient and dependent on the actions.
    - The goal predicates (road, at, capacity-predecessor) form the main checkpoints.
    - The heuristic is allowed to be non-admissible but should guide search efficiently by counting
      differences and estimating minimal required steps to fix them.
    
    Heuristic Initialization
    ------------------------
    - Extract and store the set of goal facts.
    - Partition the goal facts into subsets based on their predicate names (road, at, capacity-predecessor).
      This allows quick checking which facts from the goal are currently missing.
    - The domain has no static facts required for this heuristic.
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the current state already satisfies all goal facts; if yes, return 0.
    2. Count how many goal road facts are missing in the current state.
    3. Count how many goal capacity-predecessor facts are missing in the current state.
    4. Count how many goal at facts are missing in the current state.
    5. Consider that each missing predicate roughly requires at least one action to establish.
    6. Analyze dependencies from the domain actions:
       - To create or remove (road ?loc) implies sequences involving pick-up, drop, or drive actions.
       - Capacity-predecessor links are prerequisites for drive actions.
       - (in ?v) predicate denotes the vehicle being "in" transport; this affects counting but is transient.
    7. Heuristically assign weights to missing predicates reflecting typical minimal sequences:
       - Fixing a missing (road ?loc): at least 1 action.
       - Fixing missing (capacity-predecessor ?loc): at least 1 action.
       - Fixing missing (at ?loc1 ?loc2): at least 1-2 actions (may require drive-2 or drive sequences).
    8. Sum all weighted missing predicates to produce the heuristic estimate.
    9. Ensure the heuristic is always finite by bounding minimal costs by 1 when missing.
    
    This heuristic trades completeness for speed and effectiveness by focusing on unsatisfied goal predicates 
    and their minimal plausible action costs to satisfy.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal predicates."""
        self.goals = task.goals  # frozenset of goal facts
        # Partition goal facts by predicate for quick lookup
        self.goal_road = set()
        self.goal_capacity_pred = set()
        self.goal_at = set()
        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "road":
                self.goal_road.add(fact)
            elif pred == "capacity-predecessor":
                self.goal_capacity_pred.add(fact)
            elif pred == "at":
                self.goal_at.add(fact)

    def __call__(self, node):
        """Compute the heuristic estimate: weighted count of unsatisfied goal predicates."""
        state = node.state

        # If goal is reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Count missing goal facts by category
        missing_road = len(self.goal_road - state)
        missing_capacity_pred = len(self.goal_capacity_pred - state)
        missing_at = len(self.goal_at - state)

        # Base costs for fixing each predicate type
        # These weights reflect typical minimal action sequences inferred from domain actions:
        # - drive requires capacity-predecessor and capacity predicates
        # - pick-up and drop toggle road and capacity-predecessor predicates
        # - drive-2 moves objects at locations and toggles predicates
        # We assign 1 action per missing capacity-predecessor and missing road fact,
        # and 2 actions per missing at fact since it likely requires at least a pick-up and drop or a drive-2.

        cost_road = missing_road * 1
        cost_capacity_pred = missing_capacity_pred * 1
        cost_at = missing_at * 2

        # Add a minimal base cost to account for transitions between the required changes
        # When there are missing at facts, it likely involves additional capacity toggling; add a small overhead
        overhead = 0
        if missing_at > 0 and (missing_road > 0 or missing_capacity_pred > 0):
            overhead = 1

        heuristic_value = cost_road + cost_capacity_pred + cost_at + overhead

        return heuristic_value
