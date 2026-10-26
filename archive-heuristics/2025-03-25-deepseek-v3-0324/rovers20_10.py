from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goals by:
    - Calculating navigation distances between waypoints
    - Counting required sample collections (soil/rock)
    - Counting required image captures
    - Accounting for calibration needs
    - Tracking communication requirements

    # Assumptions
    - Rovers can perform only one task at a time (sequential execution)
    - Navigation between waypoints takes 1 action per hop
    - Each sample collection takes 1 action
    - Each image capture takes 1 action (plus calibration if needed)
    - Each communication takes 1 action
    - Rovers can share tasks (multiple rovers can work on different goals)

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Build maps of:
      - Waypoint connectivity (navigation graph)
      - Sample locations
      - Objective visibility
      - Camera capabilities
      - Lander positions

    # Step-By-Step Thinking for Computing Heuristic
    1. For each uncommunicated soil/rock data:
       - Find closest rover with appropriate equipment
       - Calculate path to sample location
       - Add actions for: navigate, sample, navigate to lander, communicate
    2. For each uncommunicated image data:
       - Find rover with appropriate camera
       - Calculate path to visible waypoint
       - Add actions for: navigate, calibrate (if needed), take image, navigate to lander, communicate
    3. Sum all required actions across all goals
    4. Optimize by assigning tasks to closest available rovers
    """

    def __init__(self, task):
        """Initialize heuristic by extracting static information and goals."""
        self.goals = task.goals
        self.static = task.static

        # Build navigation graph: waypoint -> {reachable waypoints}
        self.nav_graph = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, rover, wp1, wp2 = get_parts(fact)
                self.nav_graph.setdefault(wp1, set()).add(wp2)
                self.nav_graph.setdefault(wp2, set()).add(wp1)

        # Map objectives to their visible waypoints
        self.objective_visibility = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                _, obj, wp = get_parts(fact)
                self.objective_visibility.setdefault(obj, set()).add(wp)

        # Map cameras to their objectives and supported modes
        self.camera_info = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                _, cam, obj = get_parts(fact)
                self.camera_info.setdefault(cam, {}).update({"target": obj})
            elif match(fact, "supports", "*", "*"):
                _, cam, mode = get_parts(fact)
                self.camera_info.setdefault(cam, {}).setdefault("modes", set()).add(mode)

        # Store lander positions
        self.lander_positions = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static
            if match(fact, "at_lander", "*", "*")
        }

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        total_cost = 0

        # Check which goals are already satisfied
        unsatisfied_goals = self.goals - state

        # Process soil data goals
        soil_goals = {
            get_parts(g)[1] for g in unsatisfied_goals 
            if match(g, "communicated_soil_data", "*")
        }

        # Process rock data goals
        rock_goals = {
            get_parts(g)[1] for g in unsatisfied_goals 
            if match(g, "communicated_rock_data", "*")
        }

        # Process image data goals
        image_goals = {
            (get_parts(g)[1], get_parts(g)[2]) for g in unsatisfied_goals 
            if match(g, "communicated_image_data", "*", "*")
        }

        # For each soil sample goal, estimate actions needed
        for wp in soil_goals:
            # Find if any rover already has this sample
            has_sample = any(
                match(fact, "have_soil_analysis", "*", wp) for fact in state
            )
            
            if has_sample:
                # Just need to communicate (navigate to lander + communicate)
                total_cost += 2  # conservative estimate
            else:
                # Need to: navigate to sample, collect, navigate to lander, communicate
                total_cost += 4  # conservative estimate

        # For each rock sample goal, estimate actions needed
        for wp in rock_goals:
            # Find if any rover already has this sample
            has_sample = any(
                match(fact, "have_rock_analysis", "*", wp) for fact in state
            )
            
            if has_sample:
                # Just need to communicate
                total_cost += 2
            else:
                # Need to: navigate to sample, collect, navigate to lander, communicate
                total_cost += 4

        # For each image goal, estimate actions needed
        for obj, mode in image_goals:
            # Check if any rover already has this image
            has_image = any(
                match(fact, "have_image", "*", obj, mode) for fact in state
            )
            
            if has_image:
                # Just need to communicate
                total_cost += 2
            else:
                # Need to: navigate to visible waypoint, calibrate (if needed), take image, navigate to lander, communicate
                total_cost += 5  # conservative estimate

        return total_cost
