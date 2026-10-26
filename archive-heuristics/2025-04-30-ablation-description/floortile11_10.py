from fnmatch import fnmatch
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle the standard PDDL fact format (p obj1 obj2)
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # If it's not in the expected format, return parts by splitting whitespace
    # This might happen for object names or other non-fact strings if they appear in state/goals/static
    # but for facts like (predicate arg1 ...), the first case is correct.
    return fact_str.split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have the same number of parts as args for a meaningful match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic: # Inherit from Heuristic if available, otherwise use a basic class
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the remaining work by summing two components:
    1. The number of goal `(painted ?tile ?color)` conditions that are not
       satisfied in the current state.
    2. The number of distinct colors required by the unsatisfied goal conditions
       that are not currently held by any robot.

    # Assumptions:
    - The goal primarily consists of `(painted ?tile ?color)` facts.
    - Tiles, once painted, cannot be unpainted or repainted with a different color.
      States where a goal tile is painted with the wrong color are assumed
      to be unsolvable and are not explicitly handled (the heuristic value
      will be finite but might not reflect unsolvability).
    - The heuristic value is 0 if and only if all goal `(painted ?tile ?color)`
      conditions are met.

    # Heuristic Initialization
    - Extracts all `(painted ?tile ?color)` goal conditions from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize `unpainted_goal_count` to 0.
    2. Initialize an empty set `required_colors` to store colors needed for unpainted tiles.
    3. Iterate through each goal condition extracted during initialization (`self.goal_painted_facts`).
    4. For each goal condition `(painted ?tile ?color)`:
       a. Check if this exact fact exists in the current state (`node.state`).
       b. If the fact does *not* exist:
          i. Increment `unpainted_goal_count`.
          ii. Add `?color` to the `required_colors` set.
    5. Initialize `color_change_cost` to 0.
    6. Get the set of colors currently held by robots from the current state.
       Iterate through state facts and find all `(robot-has ?r ?c)` facts. Collect the colors `?c`.
    7. Iterate through each color `C` in the `required_colors` set.
    8. For each required color `C`:
       a. Check if any robot in the current state holds color `C`.
       b. If no robot holds color `C`, increment `color_change_cost`.
    9. The heuristic value is `unpainted_goal_count` + `color_change_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal painted facts.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        self.static = task.static

        # Extract goal conditions that are about painted tiles.
        # Goal conditions are in task.goals (a frozenset of strings).
        self.goal_painted_facts = {
            goal for goal in self.goals if match(goal, "painted", "*", "*")
        }

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        The value is the number of goal painted facts not present,
        plus a cost for required colors not held by any robot.
        """
        state = node.state  # Current world state (frozenset of strings).

        unpainted_goal_count = 0
        required_colors = set()

        # Step 3 & 4: Count unpainted goals and collect required colors
        for goal_fact in self.goal_painted_facts:
            if goal_fact not in state:
                unpainted_goal_count += 1
                # Extract color from the goal fact string
                parts = get_parts(goal_fact)
                # Ensure the fact has the expected structure before accessing parts[2]
                if len(parts) > 2:
                     required_colors.add(parts[2]) # parts[1] is tile, parts[2] is color

        # Step 6: Get colors currently held by robots
        colors_held_by_robots = set()
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                # Ensure the fact has the expected structure before accessing parts[2]
                if len(parts) > 2:
                    colors_held_by_robots.add(parts[2]) # parts[1] is robot, parts[2] is color

        # Step 5, 7 & 8: Calculate color change cost
        color_change_cost = 0
        for color in required_colors:
            if color not in colors_held_by_robots:
                 # Add 1 cost for each required color type that is not available
                 # among any robot's current color.
                 color_change_cost += 1

        # Step 9: Total heuristic value
        total_cost = unpainted_goal_count + color_change_cost

        # The heuristic is 0 iff unpainted_goal_count is 0 AND required_colors is empty
        # (which happens when unpainted_goal_count is 0).
        # So, it's 0 iff all goal painted facts are in the state.
        # This matches the requirement assuming the goal is only painted facts.

        return total_cost
