from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact matches a pattern with possible wildcards '*'.
    Example: match("(at-robot box1 location2)", "at-robot", "box1", "*") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A lightweight, domain-dependent heuristic for the 'sokoban' (secret_domain) PDDL
    variant provided. The goal facts in this domain require boxes to satisfy the
    predicate (adjacent-2 BOX). The heuristic estimates how many operator
    applications (roughly) remain until all goal adjacent-2 facts are achieved.

    Assumptions
    -----------
    - Each goal is of the form (adjacent-2 BOX). The heuristic focuses on these.
    - Achieving adjacent-2 for a single box typically requires at least one 'push'
      action (the domain's `push` operator) if the box is in the correct precondition
      state (for example, the box has the unary (at BOX) fact). If it is not,
      additional robot positioning or preparatory moves are required.
    - Robot "interaction points" are encoded in static facts of the form
      (at-robot BOX LOCATION). These indicate convenient robot locations to
      interact with that box. If the robot is already at such a location,
      fewer actions are typically required to finish that box.
    - The heuristic is not admissible; it is intended to be informative and
      computationally cheap to compute in greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal boxes (boxes that must satisfy adjacent-2).
    - Extract static facts that map a box to an interaction location via
      (at-robot BOX LOCATION) entries. This allows a small, cheap adjustment
      if the robot is already at that interaction location.
    - No other complicated static structure is required for this heuristic.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For each goal box (a box that must satisfy (adjacent-2 BOX)):
    1. If the box already has (adjacent-2 BOX) in the current state, the
       estimated cost for that box is 0.
    2. Otherwise, estimate a small integer cost depending on readily-available
       facts in the state:
       - If the state contains '(at BOX)' (the box appears to be in the special
         'at' condition used by the domain), we estimate that a single push
         is sufficient -> base cost = 1.
       - Else if the state contains any '(adjacent BOX LOCATION)' fact, we
         assume the box is near some useful location and estimate base cost = 2.
       - Otherwise estimate base cost = 3 to account for required robot moves
         and preparatory actions.
    3. If the static mapping (from initialization) provides a preferred robot
       location for interacting with this box, and the robot's current
       location (from state fact '(at-robot-2 LOCATION)') matches that preferred
       location, we lower the estimated cost by 1 (but never below 1 for an
       unfinished goal). This represents a cheap local saving when the robot
       is already well-positioned.
    4. Sum the per-box estimates to produce the final heuristic value.
    5. Guarantee that the heuristic is 0 exactly when all goals are satisfied
       (i.e., all (adjacent-2 BOX) facts are present in the state).

    Rationale
    ---------
    - The heuristic is intentionally simple and additive across goal boxes. It
      trades some accuracy for speed and robustness across the provided
      instance sizes. The adjustments using static at-robot facts make the
      heuristic more informed than a simple goals-left count by recognizing
      useful robot positioning.
    """

    def __init__(self, task):
        """
        Extract goal boxes and useful static mappings.

        Parameters:
        - task: a Task object from the planner, exposing `goals` and `static`.
        """
        # The set of goal facts (strings)
        self.goals = set(task.goals)

        # Extract which boxes appear in adjacent-2 goals.
        self.goal_boxes = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "adjacent-2" and len(parts) >= 2:
                self.goal_boxes.append(parts[1])

        # Extract static mapping (at-robot BOX LOCATION) for quick positioning hints.
        # This is a dict box -> location (string). If multiple entries exist, the
        # last one seen is kept (typically there will be at most one).
        self.box_to_robot_loc = {}
        for fact in task.static:
            if match(fact, "at-robot", "*", "*"):
                _, box, loc = get_parts(fact)
                self.box_to_robot_loc[box] = loc

    def __call__(self, node):
        """
        Compute the heuristic estimate for `node`.

        Returns an integer estimate of the number of remaining actions.
        """
        state = node.state  # frozenset of facts (strings)
        state_set = set(state)

        # If all goals already satisfied, return 0 immediately.
        if self.goals <= state_set:
            return 0

        # Determine robot's current location from facts of the form (at-robot-2 LOC).
        robot_loc = None
        for fact in state_set:
            if match(fact, "at-robot-2", "*"):
                # fact structure: (at-robot-2 locationX)
                robot_loc = get_parts(fact)[1]
                break

        total_cost = 0

        # For each goal box, estimate cost independently and sum.
        for box in self.goal_boxes:
            adj2_fact = f"(adjacent-2 {box})"
            if adj2_fact in state_set:
                continue  # cost 0 for this box

            # Base estimates based on easy-to-check state facts.
            if f"(at {box})" in state_set:
                base = 1
            else:
                # Check any adjacent facts for this box: (adjacent box LOCATION)
                has_adjacent = any(
                    match(fact, "adjacent", box, "*") for fact in state_set
                )
                base = 2 if has_adjacent else 3

            # If robot is already at the preferred interaction location for the box,
            # we can lower the cost by 1 (but do not drop below 1).
            preferred_loc = self.box_to_robot_loc.get(box)
            if preferred_loc and robot_loc and preferred_loc == robot_loc:
                base = max(1, base - 1)

            total_cost += base

        # Ensure finite integer heuristic.
        return int(total_cost)