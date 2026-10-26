from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to communicate all required data points (soil, rock, and image analyses) to the goal locations.

    # Assumptions:
    - Each rover can perform one task at a time (sample collection, image taking).
    - Navigation between visible waypoints is possible if can_traverse is true.
    - Communication requires the data to be collected and the rover to be at a visible waypoint from the lander.
    - If a rover is already at the required waypoint, no navigation is needed.

    # Heuristic Initialization
    - Extracts goal conditions (communicated data points) and static facts (can_traverse, visible, calibration targets) from the task.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each required data point (soil, rock, image):
        a. Check if the data has already been communicated.
        b. If not, determine the rover's current state and location.
        c. Estimate the actions needed to navigate to the required waypoint (if not already there).
        d. Estimate actions for sample collection or image taking if required.
        e. Estimate actions for communication.
    2. Sum the estimated actions for all required data points.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.can_traverse = set()
        self.visible = set()
        self calibration_targets = dict()
        self.on_board_cameras = dict()
        self.visible_from = dict()

        for fact in self.static:
            parts = fact[1:-1].split()
            if fact.startswith('(can_traverse '):
                r, f, t = parts[1], parts[2], parts[3]
                self.can_traverse.add((r, f, t))
            elif fact.startswith('(visible '):
                f, t = parts[1], parts[2]
                self.visible.add((f, t))
                self.visible.add((t, f))
            elif fact.startswith('(calibration_target '):
                c, o = parts[1], parts[2]
                self.calibration_targets[c] = o
            elif fact.startswith('(on_board '):
                c, r = parts[1], parts[2]
                self.on_board_cameras[c] = r
            elif fact.startswith('(visible_from '):
                o, w = parts[1], parts[2]
                self.visible_from[o] = w

    def __call__(self, node):
        """Estimate the minimum cost to achieve all required data communications."""
        state = node.state

        def get_obj(fact):
            """Extract the object from a fact string."""
            return fact[1:-1].split()[1]

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract current state information
        at_rover = {}
        at_lander = {}
        have_soil = {}
        have_rock = {}
        have_image = {}
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()

        for fact in state:
            if fact.startswith('(at '):
                r, w = fact[1:-1].split()[1], fact[1:-1].split()[2]
                at_rover[r] = w
            elif fact.startswith('(at_lander '):
                l, w = fact[1:-1].split()[1], fact[1:-1].split()[2]
                at_lander[l] = w
            elif fact.startswith('(have_soil_analysis '):
                r, w = fact[1:-1].split()[1], fact[1:-1].split()[2]
                have_soil[r] = w
            elif fact.startswith('(have_rock_analysis '):
                r, w = fact[1:-1].split()[1], fact[1:-1].split()[2]
                have_rock[r] = w
            elif fact.startswith('(have_image '):
                r, o, m = fact[1:-1].split()[1], fact[1:-1].split()[2], fact[1:-1].split()[3]
                have_image[r] = (o, m)
            elif fact.startswith('(communicated_soil_data '):
                w = fact[1:-1].split()[1]
                communicated_soil.add(w)
            elif fact.startswith('(communicated_rock_data '):
                w = fact[1:-1].split()[1]
                communicated_rock.add(w)
            elif fact.startswith('(communicated_image_data '):
                o, m = fact[1:-1].split()[1], fact[1:-1].split()[2]
                communicated_image.add((o, m))

        total_cost = 0

        # Check for soil data communication goals
        soil_goals = set()
        for goal in self.goals:
            if goal.startswith('(communicated_soil_data '):
                w = goal[1:-1].split()[1]
                soil_goals.add(w)
        for w in soil_goals:
            if w not in communicated_soil:
                # Find a rover that can collect soil from w
                # Assume one rover is available
                # Actions: navigate to w (if not already there), sample, communicate
                # Navigation: 2 actions (to and back if needed)
                # Sample: 1 action
                # Communicate: 1 action
                total_cost += 4  # conservative estimate

        # Check for rock data communication goals
        rock_goals = set()
        for goal in self.goals:
            if goal.startswith('(communicated_rock_data '):
                w = goal[1:-1].split()[1]
                rock_goals.add(w)
        for w in rock_goals:
            if w not in communicated_rock:
                # Similar to soil
                total_cost += 4

        # Check for image data communication goals
        image_goals = set()
        for goal in self.goals:
            if goal.startswith('(communicated_image_data '):
                o, m = goal[1:-1].split()[1], goal[1:-1].split()[2]
                image_goals.add((o, m))
        for (o, m) in image_goals:
            if (o, m) not in communicated_image:
                # Find a rover with imaging equipment
                # Actions: calibrate, take image, communicate
                # Navigation to waypoint with objective visible
                # Assume calibration is needed: 1 action
                # Take image: 1 action
                # Communicate: 1 action
                total_cost += 3

        return total_cost
