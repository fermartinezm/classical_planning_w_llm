from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(road locatable3)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the `secret_domain` (transport) domain.

    Summary:
    This heuristic estimates the minimum number of actions required to reach the goal state
    from a given current state. The transport domain models vehicles moving along a chain 
    of locatables with predicates describing "road", "capacity-predecessor", "capacity", "in", and "at".
    The vehicle "travels" by carrying capacity along these locatables, and the sequence of drive, pick-up, drop, and drive-2 actions progress the state.
    The heuristic thus tries to estimate how many chain moves and capacity transfers remain to meet the goal facts.

    Assumptions:
    - The domain is characterized by a chain-like structure of locatables connected by "capacity-predecessor" and "at" facts.
    - Only one "capacity" predicate exists and is manipulated by pick-up, drop, and drive actions.
    - The goal usually requires certain "road", "at", and "capacity-predecessor" predicates on locatables.
    - The heuristic does not need to be admissible but should guide greedy best-first to minimize expanded nodes efficiently.
    - Static facts are empty in this domain and thus ignored.

    Heuristic Initialization:
    - Store the goal facts (road/at/capacity-predecessor) to be achieved.
    - No static facts in this domain, so none extracted.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract the following sets from the current state:
       - road_locs: set of locatables currently having the road predicate.
       - capacity_pred_locs: set of locatables currently having capacity-predecessor.
       - at_facts: dict mapping locatable -> locatable for facts (at loc1 loc2).
       - in_locs: set of locatables currently holding the capacity ("in").

    2. Extract from the goal the set of target road locations, target capacity-predecessor locations,
       and goal (at locatable locatable) pairs.

    3. Identify differences:
       - Which locatables must gain or lose road predicate to match goal.
       - Which locatables must gain or lose capacity-predecessor predicate.
       - Which "at" relations differ from goal.

    4. Reason about capacity usage and vehicle:
       - Capacity moves along the chain; pick-up and drop actions toggle the predicates capacity, capacity-predecessor, and road on respective locatables.
       - The vehicle must be driven (drive or drive-2) to support changes.
       - The capacity must be held ("in" predicate) when driving or picking/dropping.

    5. Key observations:
       - Each change between road states at locatables likely costs 1 pick-up or drop plus intermediate drive actions.
       - Capacity-predecessor relations form a chain; adjusting these will cost at least one action per difference.
       - "at" predicates encode chains; changing them could require one or more drive-2 actions per hop.

    6. Therefore, the heuristic sums:
       - The number of mismatches in road predicate (each mismatch counts as 1 to 2 actions depending on pick/drop).
       - The number of mismatches in capacity-predecessor predicate (at least one action per mismatch).
       - The number of differing "at" relations (each difference reflects a drive-2 action to correct).
       - A small cost penalty if capacity is not held when needed (to encourage states with capacity).

    7. The total sum is an estimate of minimal actions, not necessarily admissible, but informative enough for greedy search.

    8. Returns 0 if the state is a goal state.

    """

    def __init__(self, task):
        # Store goal facts by predicate for easy lookup
        self.goal_road = set()
        self.goal_capacity_predecessor = set()
        self.goal_at = {}

        for fact in task.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "road":
                self.goal_road.add(parts[1])
            elif predicate == "capacity-predecessor":
                self.goal_capacity_predecessor.add(parts[1])
            elif predicate == "at":
                # format: (at loc1 loc2)
                self.goal_at[parts[1]] = parts[2]

        # Static facts unused in this domain
        self.static = task.static

    def __call__(self, node):
        state = node.state

        # Quickly return 0 if goal is reached
        if all(fact in state for fact in self.goal_road.union(
            f"(road {loc})" for loc in self.goal_road
        )) and all(fact in state for fact in self.goal_capacity_predecessor.union(
            f"(capacity-predecessor {loc})" for loc in self.goal_capacity_predecessor
        )) and all(f"(at {loc} {to_loc})" in state for loc, to_loc in self.goal_at.items()):
            if self.goal_reached(state):
                return 0

        # Extract current facts by predicate
        road_locs = {get_parts(f)[1] for f in state if match(f, "road", "*")}
        capacity_pred_locs = {get_parts(f)[1] for f in state if match(f, "capacity-predecessor", "*")}
        in_locs = {get_parts(f)[1] for f in state if match(f, "in", "*")}
        capacity_held = any(match(f, "capacity") for f in state)

        current_at = {}
        for f in state:
            if match(f, "at", "*", "*"):
                parts = get_parts(f)
                current_at[parts[1]] = parts[2]

        # Differences in road predicates -> cost
        road_diff = road_locs.symmetric_difference(self.goal_road)
        cost_road = len(road_diff)  # Each mismatch at least one action (pick-up or drop)

        # Differences in capacity-predecessor predicates
        capacity_pred_diff = capacity_pred_locs.symmetric_difference(self.goal_capacity_predecessor)
        cost_capacity_pred = len(capacity_pred_diff)  # Each mismatch at least one action (pick-up or drop)

        # Differences in 'at' facts
        # Count number of locatables whose 'at' target differs from goal
        at_diff = 0
        locatables_checked = set(self.goal_at.keys()) | set(current_at.keys())
        for loc in locatables_checked:
            goal_target = self.goal_at.get(loc)
            current_target = current_at.get(loc)
            if goal_target != current_target:
                at_diff += 1  # Assume one drive-2 action per difference

        # Penalize if capacity is not currently held but needed (to encourage getting capacity)
        penalty_capacity_held = 0 if capacity_held else 1

        # Sum estimated cost
        heuristic_value = cost_road + cost_capacity_pred + at_diff + penalty_capacity_held

        # Return heuristic cost estimate (finite for solvable states)
        return heuristic_value

    def goal_reached(self, state):
        """Helper: Return True if state satisfies all goals."""
        return (
            all(f"(road {loc})" in state for loc in self.goal_road)
            and all(f"(capacity-predecessor {loc})" in state for loc in self.goal_capacity_predecessor)
            and all(f"(at {loc} {to_loc})" in state for loc, to_loc in self.goal_at.items())
        )
