from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goals by:
    1. Counting unsatisfied communication goals (soil, rock, image data)
    2. Estimating the cost to:
       - Move rovers to sample locations
       - Perform sampling/imaging
       - Move to lander locations
       - Communicate data

    # Assumptions:
    - Each unsatisfied goal requires at least one action to achieve
    - Rovers can perform one task at a time (sequential actions)
    - The most efficient path is used for movement estimates
    - Sample collection and communication are atomic actions

    # Heuristic Initialization
    - Extract goal conditions and static facts
    - Build maps for waypoint visibility and rover capabilities
    - Identify lander locations

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unsatisfied communication goals:
       - Soil data not communicated for required waypoints
       - Rock data not communicated for required waypoints
       - Image data not communicated for required objectives/modes

    2. For each unsatisfied goal:
       a) If it's soil/rock data:
          - Check if rover has the analysis (count as 0 if yes)
          - Else estimate cost to:
            * Move to sample location (1 action per required move)
            * Perform sampling (1 action)
            * Move to lander-visible waypoint (1 action per required move)
            * Communicate data (1 action)

       b) If it's image data:
          - Check if rover has the image (count as 0 if yes)
          - Else estimate cost to:
            * Calibrate camera if needed (1 action)
            * Move to visible location (1 action per required move)
            * Take image (1 action)
            * Move to lander-visible waypoint (1 action per required move)
            * Communicate data (1 action)

    3. Sum all estimated actions for unsatisfied goals
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_locations = set()
        self.visible = set()
        self.can_traverse = set()
        self.calibration_targets = {}
        self.supports = {}
        self.on_board = {}
        self.equipped_for = {}

        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "at_lander", "*", "*"):
                self.lander_locations.add(parts[2])
            elif match(fact, "visible", "*", "*"):
                self.visible.add((parts[1], parts[2]))
            elif match(fact, "can_traverse", "*", "*", "*"):
                self.can_traverse.add((parts[1], parts[2], parts[3]))
            elif match(fact, "calibration_target", "*", "*"):
                self.calibration_targets[parts[1]] = parts[2]
            elif match(fact, "supports", "*", "*"):
                if parts[1] not in self.supports:
                    self.supports[parts[1]] = set()
                self.supports[parts[1]].add(parts[2])
            elif match(fact, "on_board", "*", "*"):
                self.on_board[parts[1]] = parts[2]
            elif match(fact, "equipped_for_*", "*"):
                capability = parts[0].split('_')[-1]
                if parts[1] not in self.equipped_for:
                    self.equipped_for[parts[1]] = set()
                self.equipped_for[parts[1]].add(capability)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_goals = self.goals - state

        # Process each unsatisfied goal
        for goal in unsatisfied_goals:
            parts = get_parts(goal)
            
            if match(goal, "communicated_soil_data", "*"):
                wp = parts[1]
                # Check if any rover already has this analysis
                has_analysis = any(
                    match(fact, "have_soil_analysis", "*", wp) for fact in state
                )
                if not has_analysis:
                    # Need to sample and communicate - estimate 3 actions
                    total_cost += 3  # sample (1) + move to lander (1) + communicate (1)

            elif match(goal, "communicated_rock_data", "*"):
                wp = parts[1]
                # Check if any rover already has this analysis
                has_analysis = any(
                    match(fact, "have_rock_analysis", "*", wp) for fact in state
                )
                if not has_analysis:
                    # Need to sample and communicate - estimate 3 actions
                    total_cost += 3  # sample (1) + move to lander (1) + communicate (1)

            elif match(goal, "communicated_image_data", "*", "*"):
                obj, mode = parts[1], parts[2]
                # Check if any rover already has this image
                has_image = any(
                    match(fact, "have_image", "*", obj, mode) for fact in state
                )
                if not has_image:
                    # Need to calibrate, take image, and communicate - estimate 4 actions
                    total_cost += 4  # calibrate (1) + take_image (1) + move to lander (1) + communicate (1)

        return total_cost
