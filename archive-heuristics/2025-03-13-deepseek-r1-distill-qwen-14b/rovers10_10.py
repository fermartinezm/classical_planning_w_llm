from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to communicate all required data points (soil, rock, and images) for each objective.

    # Assumptions:
    - Each objective may require multiple types of data (soil, rock, high-res, low-res, colour images).
    - Rovers can navigate between waypoints if they are visible and connected.
    - Communication requires the rover to be at a waypoint visible to the lander.
    - Imaging requires calibration and visibility between the objective and the waypoint.

    # Heuristic Initialization
    - Extracts static facts about waypoint connectivity, calibration targets, and communication visibility.
    - Maps each objective to its required data types.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each objective in the goal:
        a. If the required data type is already communicated, skip.
        b. If not, find the nearest waypoint that can communicate the data.
        c. Calculate the steps needed to navigate to that waypoint.
        d. If imaging is required, ensure the rover is calibrated and has the necessary equipped camera.
        e. Sum the actions for navigation, calibration (if needed), image capture, and communication.
    2. Sum the actions for all objectives to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts and goal information."""
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.waypoint_graph = self._build_waypoint_graph(static_facts)
        self.calibration_targets = self._extract_calibration_targets(static_facts)
        self.visible_waypoints = self._extract_visible_waypoints(static_facts)
        self.communication_visibilities = self._extract_communication_visibilities(static_facts)

        # Map each objective to its required data types
        self.objective_data_requirements = {}
        for goal in self.goals:
            predicate, obj, data_type = get_parts(goal)
            if predicate == "communicated_image_data":
                if obj not in self.objective_data_requirements:
                    self.objective_data_requirements[obj] = set()
                self.objective_data_requirements[obj].add(data_type)
            elif predicate == "communicated_soil_data" or predicate == "communicated_rock_data":
                obj = obj
                data_type = predicate.split('_')[1]
                if obj not in self.objective_data_requirements:
                    self.objective_data_requirements[obj] = set()
                self.objective_data_requirements[obj].add(data_type)

    def __call__(self, node):
        """Compute the estimated number of actions to achieve the goal."""
        state = node.state
        current_waypoint = self._get_rover_position(state)
        total_actions = 0

        # For each objective, check if all required data types are communicated
        for obj, required_data in self.objective_data_requirements.items():
            # Check if all required data types are already communicated
            communicated_data = set()
            for fact in state:
                if fact.startswith("(communicated_"):
                    data_type = fact.split('_')[2]
                    if obj in fact:
                        communicated_data.add(data_type)
            if communicated_data.issuperset(required_data):
                continue

            # Find the nearest waypoint that can communicate the required data
            communication_waypoints = []
            for waypoint in self.visible_waypoints.get(obj, []):
                if waypoint in self.communication_visibilities:
                    communication_waypoints.append(waypoint)
            if not communication_waypoints:
                continue  # No possible way to communicate this data

            # Choose the nearest waypoint (simplified as the first one for this example)
            target_waypoint = communication_waypoints[0]

            # Calculate navigation actions
            path = self._find_path(current_waypoint, target_waypoint)
            if path:
                total_actions += len(path)  # Each move is one action

            # Check if calibration is needed
            camera_mode = None
            for fact in state:
                if fact.startswith("(have_image rover1 ") and obj in fact:
                    parts = get_parts(fact)
                    camera_mode = parts[3]
                    break
            if not camera_mode:
                # Need to calibrate
                total_actions += 2  # calibrate and take_image actions

            # Communication action
            total_actions += 1  # communicate_data action

        return total_actions

    def _build_waypoint_graph(self, static_facts):
        """Build a graph of waypoints that can be traversed."""
        graph = {}
        for fact in static_facts:
            if fact.startswith("(can_traverse "):
                _, rover, from_wpt, to_wpt = get_parts(fact)
                if from_wpt not in graph:
                    graph[from_wpt] = set()
                graph[from_wpt].add(to_wpt)
        return graph

    def _extract_calibration_targets(self, static_facts):
        """Extract calibration targets for each camera."""
        targets = {}
        for fact in static_facts:
            if fact.startswith("(calibration_target "):
                _, camera, obj = get_parts(fact)
                if obj not in targets:
                    targets[obj] = set()
                targets[obj].add(camera)
        return targets

    def _extract_visible_waypoints(self, static_facts):
        """Extract waypoints visible from each objective."""
        visible = {}
        for fact in static_facts:
            if fact.startswith("(visible_from "):
                _, obj, wpt = get_parts(fact)
                if obj not in visible:
                    visible[obj] = set()
                visible[obj].add(wpt)
        return visible

    def _extract_communication_visibilities(self, static_facts):
        """Extract waypoints visible to the lander for communication."""
        visible = set()
        for fact in static_facts:
            if fact.startswith("(visible ") and "lander" in fact:
                _, from_wpt, to_wpt = get_parts(fact)
                visible.add((from_wpt, to_wpt))
        return visible

    def _find_path(self, start, end):
        """Find a path from start to end using BFS."""
        visited = set()
        queue = [(start, [start])]
        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.waypoint_graph.get(current, []):
                if neighbor not in path:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        return None

    def _get_rover_position(self, state):
        """Get the current waypoint of the rover."""
        for fact in state:
            if fact.startswith("(at rover1 "):
                return get_parts(fact)[2]
        return None

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
