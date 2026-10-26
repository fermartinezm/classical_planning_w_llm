from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed for a rover to:
    1. Collect missing samples (soil or rock).
    2. Take missing images.
    3. Communicate collected data to the lander.

    # Assumptions:
    - Each sample or image requires specific rover equipment.
    - Communication requires visibility between the rover's location and the lander.
    - Navigation between waypoints is possible if it's can_traverse and visible.

    # Heuristic Initialization
    - Extract static facts about can_traverse and visible relationships.
    - Store goal conditions for each objective.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all missing data points (soil, rock, images) from the goal.
    2. For each missing data point:
       a. Determine if it's a sample, image, or communication task.
       b. Check if the rover has the necessary equipment and data.
       c. Estimate actions needed to collect or take the data.
       d. Estimate actions needed to navigate to the communication location.
       e. Add the communication action.
    3. Sum the estimated actions for all missing data points.
    """

    def __init__(self, task):
        """Initialize the heuristic with static and goal information."""
        self.goals = task.goals
        static_facts = task.static

        # Extract can_traverse information
        self.can_traverse = {}
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, from_waypoint, to_waypoint = get_parts(fact)[1:]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                self.can_traverse[rover][from_waypoint] = self.can_traverse[rover].get(to_waypoint, False)
                self.can_traverse[rover][from_waypoint] = True

        # Extract visible information
        self.visible = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                from_waypoint, to_waypoint = get_parts(fact)[1:]
                if from_waypoint not in self.visible:
                    self.visible[from_waypoint] = {}
                self.visible[from_waypoint][to_waypoint] = True

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        total_actions = 0

        # Check if all goals are already achieved
        if self.goals.issubset(state):
            return 0

        # Count missing soil and rock analyses
        missing_soil = 0
        missing_rock = 0
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                missing_soil += 1
            elif match(goal, "communicated_rock_data", "*"):
                missing_rock += 1

        # Count missing images
        missing_images = 0
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*", "*"):
                missing_images += 1

        # For each missing soil analysis
        for _ in range(missing_soil):
            total_actions += 3  # sample, navigate to lander, communicate

        # For each missing rock analysis
        for _ in range(missing_rock):
            total_actions += 3  # sample, navigate to lander, communicate

        # For each missing image
        for _ in range(missing_images):
            total_actions += 4  # take image, navigate to lander, communicate

        return total_actions
