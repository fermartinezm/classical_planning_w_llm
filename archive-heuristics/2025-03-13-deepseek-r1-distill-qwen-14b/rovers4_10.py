from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goal conditions, including:
    - Communicating soil and rock analysis data.
    - Communicating image data in different modes (colour, high_res, low_res).

    # Assumptions:
    - Each rover can carry one sample at a time.
    - Communication requires visibility between the rover's location and the lander.
    - Image data requires calibration and visibility from the objective.

    # Heuristic Initialization
    - Extracts goal locations for each objective.
    - Stores static facts about waypoints, their visibility, and traversal capabilities.

    # Step-by-Step Thinking for Computing Heuristic
    1. Check if all goal conditions are already met. If so, return 0.
    2. For each objective (soil, rock, image):
        a. If the data is already communicated, skip.
        b. If the data is not communicated, calculate the required actions:
            - For soil/rock: Navigation to the sample, collection, and communication.
            - For images: Calibration, image capture, and communication.
    3. Sum the costs for all objectives to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal information and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each objective
        self.goal_communications = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'communicated_soil_data':
                waypoint = parts[1]
                self.goal_communications[waypoint] = ('soil', waypoint)
            elif parts[0] == 'communicated_rock_data':
                waypoint = parts[1]
                self.goal_communications[waypoint] = ('rock', waypoint)
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                self.goal_communications[(obj, mode)] = ('image', obj, mode)

        # Preprocess static facts
        self.waypoint_graph = {}
        self.waypoint_visible_from = {}
        self.waypoint_visible_to = {}
        self.waypoint_calibrated = {}
        for fact in self.static:
            if fact.startswith('(can_traverse'):
                rover, from_w, to_w = fact[1:-1].split()
                if rover not in self.waypoint_graph:
                    self.waypoint_graph[rover] = {}
                self.waypoint_graph[rover][from_w] = self.waypoint_graph[rover].get(to_w, 0) + 1
            elif fact.startswith('(visible'):
                from_w, to_w = fact[1:-1].split()
                if from_w not in self.waypoint_visible_from:
                    self.waypoint_visible_from[from_w] = set()
                self.waypoint_visible_from[from_w].add(to_w)
                if to_w not in self.waypoint_visible_to:
                    self.waypoint_visible_to[to_w] = set()
                self.waypoint_visible_to[to_w].add(from_w)
            elif fact.startswith('(calibration_target'):
                camera, obj = fact[1:-1].split()
                self.waypoint_calibrated[camera] = obj

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state
        if self.is_goal_state(state):
            return 0

        heuristic_cost = 0

        # Check communicated_soil_data
        for waypoint in self.goal_communications:
            if isinstance(waypoint, str):
                if not self.is_communicated_soil(state, waypoint):
                    heuristic_cost += self.get_communicate_cost('soil', waypoint)

        # Check communicated_rock_data
        for waypoint in self.goal_communications:
            if isinstance(waypoint, str):
                if not self.is_communicated_rock(state, waypoint):
                    heuristic_cost += self.get_communicate_cost('rock', waypoint)

        # Check communicated_image_data
        for (obj, mode) in self.goal_communications:
            if not self.is_communicated_image(state, obj, mode):
                heuristic_cost += self.get_communicate_cost('image', obj, mode)

        return heuristic_cost

    def is_goal_state(self, state):
        """Check if all goal conditions are met."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True

    def get_communicate_cost(self, goal_type, *args):
        """Calculate the cost to communicate a specific data type."""
        cost = 0
        if goal_type == 'soil':
            waypoint = args[0]
            cost += 2  # Sample and communicate
        elif goal_type == 'rock':
            waypoint = args[0]
            cost += 2  # Sample and communicate
        elif goal_type == 'image':
            obj, mode = args
            cost += 3  # Calibrate, take image, communicate
        return cost

    def is_communicated_soil(self, state, waypoint):
        """Check if soil data for a waypoint is communicated."""
        return f"(communicated_soil_data {waypoint})" in state

    def is_communicated_rock(self, state, waypoint):
        """Check if rock data for a waypoint is communicated."""
        return f"(communicated_rock_data {waypoint})" in state

    def is_communicated_image(self, state, obj, mode):
        """Check if image data for an objective and mode is communicated."""
        return f"(communicated_image_data {obj} {mode})" in state
