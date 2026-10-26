from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to communicate all required data (soil, rock, and image) to the lander.

    # Assumptions:
    - Each rover can collect soil, rock, and image data.
    - Communication of each data type requires the rover to be at a waypoint visible to the lander.
    - Moving between waypoints takes a fixed number of actions.
    - Communicating each data point takes a fixed number of actions.

    # Heuristic Initialization
    - Extract the goal conditions to determine which data needs to be communicated.
    - Extract static facts to identify communication points and visible waypoints.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the number of soil, rock, and image data points that need to be communicated.
    2. For each data point:
       a. If the data is already communicated, no actions are needed.
       b. If the data is collected but not communicated, estimate the actions needed to move to the communication point and communicate the data.
       c. If the data is not collected, estimate the actions needed to collect the data and then move to the communication point to communicate it.
    3. Sum the estimated actions for all missing data points to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract communication points (waypoints visible to the lander)
        self.communication_points = set()
        for fact in self.static:
            if fnmatch(fact, '(visible ?w - waypoint ?p - waypoint)'):
                parts = fact[1:-1].split()
                if parts[2] == 'lander':
                    self.communication_points.add(parts[1])

        # Extract waypoints that are visible from objectives (used for imaging)
        self.visible_from_objectives = set()
        for fact in self.static:
            if fnmatch(fact, '(visible_from ?o - objective ?w - waypoint)'):
                self.visible_from_objectives.add(fact)

    def __call__(self, node):
        """Estimate the minimum cost to achieve the goal state."""
        state = node.state
        goal_communicated = set()

        # Check which goals are already achieved
        for goal in self.goals:
            if goal in state:
                goal_communicated.add(goal)

        # Count remaining communications needed
        soil_needed = 0
        rock_needed = 0
        image_needed = 0

        # Check soil communication goals
        for goal in self.goals:
            if fnmatch(goal, '(communicated_soil_data ?w - waypoint)'):
                if not any(fnmatch(f, '(communicated_soil_data %s)' % goal.split()[1]) for f in state):
                    soil_needed += 1

        # Check rock communication goals
        for goal in self.goals:
            if fnmatch(goal, '(communicated_rock_data ?w - waypoint)'):
                if not any(fnmatch(f, '(communicated_rock_data %s)' % goal.split()[1]) for f in state):
                    rock_needed += 1

        # Check image communication goals
        for goal in self.goals:
            if fnmatch(goal, '(communicated_image_data ?o - objective ?m - mode)'):
                if not any(fnmatch(f, '(communicated_image_data %s %s)' % (goal.split()[1], goal.split()[2])) for f in state):
                    image_needed += 1

        total_cost = 0

        # Soil and rock sample handling
        for _ in range(soil_needed + rock_needed):
            total_cost += 4  # Assume 2 steps to collect and 2 to move to communication point

        # Image handling
        for _ in range(image_needed):
            total_cost += 4  # Assume 2 steps to take image and 2 to move to communication point

        # Communication steps
        total_cost += (soil_needed + rock_needed + image_needed) * 2

        return total_cost
