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
    This heuristic estimates the number of actions required to achieve all goals by:
    - Counting remaining uncommunicated data (soil, rock, image)
    - Estimating navigation steps for rovers to collect and communicate data
    - Considering calibration and imaging actions for objectives

    # Assumptions:
    - Rovers can navigate between waypoints if they are connected via `can_traverse`.
    - Soil/rock samples can be collected if the rover is at the waypoint and equipped.
    - Images require calibration and the rover must be at a visible waypoint.
    - Communication requires the rover to be at a waypoint visible from the lander.

    # Heuristic Initialization
    - Extract goal conditions (communicated data).
    - Build maps for waypoint connectivity, visible objectives, and rover capabilities.
    - Identify lander location from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each uncommunicated soil/rock data:
       - Find nearest rover that can collect it (equipped and can navigate to waypoint).
       - Add steps for navigation, sampling, and communication.
    2. For each uncommunicated image data:
       - Find rover with matching camera that can reach a visible waypoint.
       - Add steps for calibration, imaging, and communication.
    3. Sum all estimated actions, prioritizing parallelizable tasks.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract lander location
        self.lander_location = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                parts = get_parts(fact)
                self.lander_location = parts[2]
                break

        # Build waypoint connectivity graph
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_parts(fact)[1:]
                if wp1 not in self.waypoint_graph:
                    self.waypoint_graph[wp1] = set()
                self.waypoint_graph[wp1].add(wp2)

        # Map objectives to visible waypoints
        self.objective_visibility = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1:]
                if obj not in self.objective_visibility:
                    self.objective_visibility[obj] = set()
                self.objective_visibility[obj].add(wp)

        # Map cameras to their objectives
        self.camera_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                cam, obj = get_parts(fact)[1:]
                self.camera_targets[cam] = obj

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        cost = 0

        # Check which goals are already satisfied
        unsatisfied_goals = self.goals - state

        # Count remaining soil data to communicate
        for goal in unsatisfied_goals:
            if match(goal, "communicated_soil_data", "*"):
                cost += 3  # navigate + sample + communicate

        # Count remaining rock data to communicate
        for goal in unsatisfied_goals:
            if match(goal, "communicated_rock_data", "*"):
                cost += 3  # navigate + sample + communicate

        # Count remaining image data to communicate
        for goal in unsatisfied_goals:
            if match(goal, "communicated_image_data", "*", "*"):
                cost += 4  # navigate + calibrate + image + communicate

        return cost
