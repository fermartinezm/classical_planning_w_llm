from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot-2 location2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' (Sokoban-like) domain.

    Summary
    -------
    This heuristic estimates the minimum number of required actions to reach the goal state
    by focusing on how many boxes have already achieved the goal predicate `(adjacent-2 box)`
    and how many still need to. It considers that each box must be pushed so that it satisfies
    `(adjacent-2 box)`. The heuristic estimates remaining pushes and approximates the moves 
    the robot must make to get into position to push each box to its goal.

    Assumptions
    -----------
    - The goal is to have all boxes satisfy `(adjacent-2 box)`.
    - Each box that is not yet adjacent-2 requires at least one push action.
    - Moving the robot to the required location for pushing costs at least one move-2 or push-2 action.
    - The robot can only push one box at a time.
    - The predicates `(at-robot-2 location)` give the current robot position.
    - The adjacency info `(adjacent box location)` and `(clear location1 location2)` represent possible moves.
    - Heuristic is not guaranteed to be admissible but should guide greedy best-first search efficiently.

    Heuristic Initialization
    ------------------------
    - Extract the goal set: all boxes must have `(adjacent-2 box)`.
    - Extract static adjacency and clear relations from static facts to build a connectivity graph of locations.
    - Extract all box objects and location objects for bookkeeping.
    - Prepare lookup structures to quickly assess box locations and robot location in any state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify boxes that are already adjacent-2 (goal satisfied) and those that are not.
    2. For each box not adjacent-2, estimate cost:
       a) At least 1 push (to achieve adjacent-2).
       b) At least 1 robot movement (move-2 or push-2) to position the robot properly for pushing.
    3. Approximate the robot movements:
       - Check current robot location `(at-robot-2 loc)`.
       - For each box, identify its current location from facts `(adjacent box location)` or `(at box)`.
       - Since exact path distances are unknown (no full map given), assume 1 movement of the robot to reach a suitable pushing location per box.
    4. The heuristic sums the pushes needed (number of boxes not adjacent-2) plus the robot moves needed.
    5. If no box is left to push, the heuristic is zero (goal reached).
    6. This heuristic is efficiently computable and informative enough to reduce node expansions.

    Note
    ----
    The heuristic does not attempt complex shortest-path computations or check blockages; it stays computationally light.
    """

    def __init__(self, task):
        """Initialize by extracting static predicates and goal box set."""
        self.goals = task.goals

        # Extract the set of boxes that must achieve (adjacent-2 box) in the goal
        self.goal_boxes = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "adjacent-2" and len(parts) == 2:
                self.goal_boxes.add(parts[1])

        # Extract static adjacency relations between locations for robot moves
        # and clear facts for extra connectivity info (not fully used but stored for possible extension)
        self.location_adjacents = dict()  # loc -> set of adjacent locations (via clear)
        self.box_adjacents = dict()  # box -> set of adjacent locations (from "adjacent" predicate)

        # Set of all known locations
        self.locations = set()

        # Scan static facts to build:
        # - location adjacency graph from (clear ?loc1 ?loc2)
        # - box adjacency info from (adjacent box location)
        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) >= 3:
                pred = parts[0]
                if pred == "clear" and len(parts) == 3:
                    loc1, loc2 = parts[1], parts[2]
                    self.locations.add(loc1)
                    self.locations.add(loc2)
                    self.location_adjacents.setdefault(loc1, set()).add(loc2)
                elif pred == "adjacent" and len(parts) == 3:
                    box, loc = parts[1], parts[2]
                    self.box_adjacents.setdefault(box, set()).add(loc)
                    self.locations.add(loc)

        # It's possible some locations appear only in robot or box predicates; add them anyway
        for fact in task.static:
            parts = get_parts(fact)
            for part in parts[1:]:
                if part.startswith("location"):
                    self.locations.add(part)

        # Prepare a quick method to detect boxes from goal_boxes plus keys in box_adjacents
        self.all_boxes = set(self.box_adjacents.keys()) | self.goal_boxes

    def __call__(self, node):
        state = node.state

        # If goal achieved, heuristic = 0
        if self.goals <= state:
            return 0

        # Determine current robot location (at-robot-2 ?loc)
        robot_loc = None
        for fact in state:
            if fact.startswith("(at-robot-2 "):
                # fact format: (at-robot-2 locationX)
                parts = get_parts(fact)
                if len(parts) == 2:
                    robot_loc = parts[1]
                    break

        # Build mapping box -> its location via (adjacent box location) or fallback (at box) as fallback location
        box_current_locs = dict()

        # We prioritize (adjacent box loc) to determine box location; if none, check (at box)
        for box in self.all_boxes:
            found = False
            for fact in state:
                parts = get_parts(fact)
                if len(parts) == 3:
                    pred = parts[0]
                    if pred == "adjacent" and parts[1] == box:
                        box_current_locs[box] = parts[2]
                        found = True
                        break
            if not found:
                # fallback: check (at box)
                for fact in state:
                    parts = get_parts(fact)
                    if len(parts) == 2 and parts[0] == "at" and parts[1] == box:
                        # The 'at' predicate in the domain file only has one argument? Actually domain-file shows '(at ?v1 - box)'
                        # So only one param? Then location unknown here, we use None to mark unknown.
                        box_current_locs[box] = None
                        found = True
                        break
            if not found:
                # No info found, set None
                box_current_locs[box] = None

        # Count how many boxes are already adjacent-2 (goal satisfied)
        satisfied_boxes = 0
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "adjacent-2" and parts[1] in self.goal_boxes:
                satisfied_boxes += 1

        # Calculate how many boxes still need pushing
        boxes_remaining = len(self.goal_boxes) - satisfied_boxes
        if boxes_remaining <= 0:
            # In case goal partial, heuristic=0
            return 0

        # Estimate cost per remaining box:
        # - at least 1 push action to reach adjacent-2
        # - at least 1 robot move to get ready for pushing (approximate)
        # So cost per box >= 2 actions

        # More precise information may improve heuristic but is expensive; we keep it simple but effective.

        # If robot location is unknown, assume minimal robot move cost per box is 1
        robot_move_per_box = 1

        # Total heuristic cost
        estimated_cost = boxes_remaining * (1 + robot_move_per_box)  # push + robot move per box

        # Return finite heuristic cost estimate > 0 when not in goal
        return estimated_cost
