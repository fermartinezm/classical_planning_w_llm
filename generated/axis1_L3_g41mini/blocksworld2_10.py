from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by stripping parentheses and splitting by space.

    Example:
        "(at locatable1 locatable2)" -> ["at", "locatable1", "locatable2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the "secret_domain" (transport domain).

    Summary:
    This heuristic estimates the number of actions required to reach the goal state, 
    based primarily on the following aspects of the state:
    - Which objects currently have the "road" predicate.
    - Which objects currently have "capacity-predecessor".
    - Which objects are "in" (some vehicle or entity).
    - The "at" relation between objects.

    The domain models entities with relationships between "locatables" connected by "road" relations, 
    with "capacity-predecessor" and "capacity" predicates controlling the applicability of actions "drive", 
    "pick-up", "drop", and "drive-2".

    Assumptions:
    - The goal consists of conditions on "road", "at", and "capacity-predecessor" predicates on locatables.
    - Actions modify these predicates allowing movement and capacity changes.
    - The heuristic approximates the number of needed "drive", "pick-up", "drop" and "drive-2" actions 
      primarily by counting discrepancies between the current and goal facts.
    - It assumes transport/movement actions affect primarily the objects identified by "road", "capacity-predecessor", and "at".

    Heuristic Initialization:
    - Extract the goal predicates, dividing them by predicate types: goal roads, goal at-relations, and goal capacity-predecessors.
    - No static facts are used since the domain does not specify any.
    
    Step-By-Step Thinking for Computing Heuristic:
    1. Extract from the current state:
       - Objects that currently have "road"
       - Objects that currently have "capacity-predecessor"
       - Objects that are "in"
       - Current "at" relations between objects stored as (obj, loc).
    2. Compare these to the goal state goals:
       - Count how many "road" predicates in the goal are currently missing.
       - Count how many "capacity-predecessor" predicates in the goal are currently missing.
       - Count how many "at" relations in the goal differ from current state.
    3. Additional observations:
       - If some objects required to be "road" in the goal are currently not "road", this likely requires at least a "drop" or a "drive" action.
       - Differences in "at" relations require actions to move objects between locations.
       - Changes in "capacity-predecessor" are related to carrying capacity/predecessor relations and may require "pick-up" or "drop" actions.
    4. The heuristic sums:
       - The count of missing goal "road" predicates (each likely needing an action).
       - The count of missing "capacity-predecessor" predicates (each likely requiring an action).
       - The number of mismatching "at" relations between the current and goal states.
       - Plus a penalty if any objects are currently "in" (indicating carrying state requiring action).
    5. This count approximates necessary transport and capacity management actions in the domain.
    6. The heuristic returns 0 if and only if all goal predicates hold (goal state).

    The heuristic is not guaranteed admissible but attempts to closely estimate action counts for guiding greedy best-first search.
    """

    def __init__(self, task):
        """Initialize by parsing goal predicates."""
        self.goal_roads = set()
        self.goal_capacity_preds = set()
        self.goal_at_pairs = dict()  # obj -> location

        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            if pred == "road":
                self.goal_roads.add(parts[1])
            elif pred == "capacity-predecessor":
                self.goal_capacity_preds.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                self.goal_at_pairs[parts[1]] = parts[2]

        # No static info is used in this heuristic for this domain.

    def __call__(self, node):
        """Estimate cost to goal from the current state."""
        state = node.state

        # Extract current facts of interest:
        current_roads = set()
        current_capacity_preds = set()
        current_in = set()
        current_at = dict()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "road":
                current_roads.add(parts[1])
            elif pred == "capacity-predecessor":
                current_capacity_preds.add(parts[1])
            elif pred == "in":
                # The predicate "in" has a single parameter according to domain (e.g. "(in v1)"),
                # this means the object is "in".
                # We'll count these to penalize carrying state.
                current_in.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                current_at[parts[1]] = parts[2]

        # Count missing roads in the current state compared to goal:
        missing_roads = len(self.goal_roads - current_roads)

        # Count missing capacity-predecessor in current state compared to goal:
        missing_capacity_preds = len(self.goal_capacity_preds - current_capacity_preds)

        # Count mismatches in "at" predicate between current and goal.
        # Objects appearing in goal_at_pairs but not at the correct location count as 1 each.
        wrong_at = 0
        for obj, goal_loc in self.goal_at_pairs.items():
            if obj not in current_at or current_at[obj] != goal_loc:
                wrong_at += 1

        # Penalize objects that are currently "in" as they likely require drop or pick-up actions to align with goal.
        in_penalty = len(current_in)

        # Combine the counts as heuristic estimate.
        # Weighted sum: each missing or wrong fact counts as one action estimate,
        # plus an additional penalty for carried objects.
        heuristic_value = missing_roads + missing_capacity_preds + wrong_at + in_penalty

        # If the heuristic value is zero, we are in a goal state.
        return heuristic_value
