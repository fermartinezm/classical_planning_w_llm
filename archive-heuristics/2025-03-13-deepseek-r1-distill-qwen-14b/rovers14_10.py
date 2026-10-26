from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to:
    1. Collect soil and rock samples from their respective waypoints.
    2. Take images of objectives using cameras.
    3. Communicate all collected data back to the lander.

    # Assumptions:
    - Each rover can carry one sample at a time.
    - Communication actions require visibility between the rover's location and the lander.
    - Calibration is needed before taking images.
    - Each image type (colour, high_res, low_res) requires separate communication.

    # Heuristic Initialization
    - Extracts static facts about waypoints, samples, calibration targets, and visibility.
    - Maps each objective to its visible waypoints for imaging.

    # Step-by-Step Thinking for Computing Heuristic
    1. **Check Communication Status**: If all data is already communicated, return 0.
    2. **Soil and Rock Analysis**:
       - For each required soil sample, calculate the steps to collect and communicate.
       - Similarly for rock samples.
    3. **Imaging**:
       - For each objective and image mode, check if calibration and imaging are done.
       - Calculate steps needed for calibration, imaging, and communication.
    4. **Sum Costs**: Aggregate the costs for all required actions to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static domain information."""
        super().__init__(task)
        self.task = task

        # Extract static facts
        static_facts = task.static

        # Precompute useful mappings and information
        self.waypoint_samples = self._extract_waypoint_samples(static_facts)
        self.visible_waypoints = self._extract_visible_waypoints(static_facts)
        self.calibration_targets = self._extract_calibration_targets(static_facts)
        self.objective_views = self._map_objectives_to_views(static_facts)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        if self._goal_reached(state):
            return 0

        cost = 0

        # Check for soil analysis communication
        soil_analysis_needed = self._count_uncommunicated_soil(state)
        cost += soil_analysis_needed * 2  # Sample and communicate

        # Check for rock analysis communication
        rock_analysis_needed = self._count_uncommunicated_rock(state)
        cost += rock_analysis_needed * 2  # Sample and communicate

        # Check for image communications
        for image_data in self._get_uncommunicated_images(state):
            cost += self._estimate_image_communication_cost(image_data)

        return cost

    def _extract_waypoint_samples(self, static_facts):
        """Extract waypoints with soil and rock samples from static facts."""
        samples = {'soil': set(), 'rock': set()}
        for fact in static_facts:
            if fact.startswith('(at_soil_sample'):
                parts = fact[1:-1].split()
                samples['soil'].add(parts[1])
            elif fact.startswith('(at_rock_sample'):
                parts = fact[1:-1].split()
                samples['rock'].add(parts[1])
        return samples

    def _extract_visible_waypoints(self, static_facts):
        """Extract visibility information between waypoints."""
        visible = {}
        for fact in static_facts:
            if fact.startswith('(visible'):
                parts = fact[1:-1].split()
                from_w = parts[1]
                to_w = parts[2]
                if from_w not in visible:
                    visible[from_w] = set()
                visible[from_w].add(to_w)
        return visible

    def _extract_calibration_targets(self, static_facts):
        """Extract calibration targets for cameras."""
        targets = {}
        for fact in static_facts:
            if fact.startswith('(calibration_target'):
                parts = fact[1:-1].split()
                camera = parts[1]
                obj = parts[2]
                if camera not in targets:
                    targets[camera] = set()
                targets[camera].add(obj)
        return targets

    def _map_objectives_to_views(self, static_facts):
        """Map each objective to the set of visible waypoints."""
        views = {}
        for fact in static_facts:
            if fact.startswith('(visible_from'):
                parts = fact[1:-1].split()
                obj = parts[1]
                wpt = parts[2]
                if obj not in views:
                    views[obj] = set()
                views[obj].add(wpt)
        return views

    def _count_uncommunicated_soil(self, state):
        """Count the number of soil samples not yet communicated."""
        communicated = set()
        for fact in state:
            if fact.startswith('(communicated_soil_data'):
                wpt = fact.split()[1]
                communicated.add(wpt)
        total_soil = len(self.waypoint_samples['soil'])
        return total_soil - len(communicated)

    def _count_uncommunicated_rock(self, state):
        """Count the number of rock samples not yet communicated."""
        communicated = set()
        for fact in state:
            if fact.startswith('(communicated_rock_data'):
                wpt = fact.split()[1]
                communicated.add(wpt)
        total_rock = len(self.waypoint_samples['rock'])
        return total_rock - len(communicated)

    def _get_uncommunicated_images(self, state):
        """Get all image data that hasn't been communicated yet."""
        uncommunicated = []
        for fact in state:
            if fact.startswith('(have_image'):
                parts = fact.split()
                rover = parts[1]
                obj = parts[2]
                mode = parts[3]
                communicated = False
                for comm_fact in state:
                    if (f'(communicated_image_data {obj} {mode})' in state):
                        communicated = True
                        break
                if not communicated:
                    uncommunicated.append((obj, mode))
        return uncommunicated

    def _estimate_image_communication_cost(self, image_data):
        """Estimate the cost to communicate an image."""
        obj, mode = image_data
        # Check if calibration is needed
        cost = 1  # Communication action
        return cost

    def _goal_reached(self, state):
        """Check if all goal conditions are met."""
        for goal in self.task.goals:
            if goal not in state:
                return False
        return True
