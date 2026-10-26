from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to collect and communicate all required data points.

    # Assumptions:
    - Each rover can move between visible waypoints.
    - A rover must be equipped with the appropriate tools to collect soil, rock, or image data.
    - Image data requires calibration before it can be taken.
    - Data communication requires the rover to be at a waypoint visible to the lander.

    # Heuristic Initialization
    - Extracts static facts about waypoints, rover equipment, and calibration targets.
    - Maps waypoints to their visible neighbors and whether they are equipped for specific tasks.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the number of each type of data (soil, rock, image) that needs to be communicated.
    2. For each data type, determine if the rover is already in a position to collect it.
    3. If calibration is needed, add the cost of calibrating the camera.
    4. Calculate the number of navigation actions required to reach the necessary waypoints.
    5. Sum the costs for all required actions to estimate the total number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information into useful data structures
        self.visible_waypoints = self._extract_visible_waypoints(static_facts)
        self.waypoint_equipment = self._extract_waypoint_equipment(static_facts)
        self.calibration_targets = self._extract_calibration_targets(static_facts)
        self.rover_equipment = self._extract_rover_equipment(task, static_facts)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Count remaining data points
        soil_data = self._count_remaining_data(state, 'communicated_soil_data')
        rock_data = self._count_remaining_data(state, 'communicated_rock_data')
        image_data = self._count_remaining_data(state, 'communicated_image_data')

        total_actions = 0

        # Handle soil data
        if soil_data > 0:
            total_actions += soil_data * 2  # Sample and communicate

        # Handle rock data
        if rock_data > 0:
            total_actions += rock_data * 2  # Sample and communicate

        # Handle image data
        if image_data > 0:
            # Check if calibration is needed
            if not self._is_calibrated(state):
                total_actions += 1  # Calibrate
            # Each image requires taking and communicating
            total_actions += image_data * 2

        return total_actions

    def _extract_visible_waypoints(self, static_facts):
        """Extract which waypoints are visible from each other."""
        visible = {}
        for fact in static_facts:
            if fnmatch(fact, '(visible ?w ?p)'):
                w, p = fact[9:-1].split()
                if w not in visible:
                    visible[w] = []
                visible[w].append(p)
        return visible

    def _extract_waypoint_equipment(self, static_facts):
        """Extract which waypoints are equipped for soil or rock analysis."""
        equipped = {'soil': {}, 'rock': {}}
        for fact in static_facts:
            if fnmatch(fact, '(at_soil_sample ?w)'):
                equipped['soil'][fact[12:-1]] = True
            elif fnmatch(fact, '(at_rock_sample ?w)'):
                equipped['rock'][fact[13:-1]] = True
        return equipped

    def _extract_calibration_targets(self, static_facts):
        """Extract calibration targets for each camera."""
        targets = {}
        for fact in static_facts:
            if fnmatch(fact, '(calibration_target ?i ?o)'):
                i, o = fact[17:-1].split()
                if i not in targets:
                    targets[i] = []
                targets[i].append(o)
        return targets

    def _extract_rover_equipment(self, task, static_facts):
        """Extract which rovers are equipped with which tools."""
        equipment = {}
        for rover in task.objects.get('rover', []):
            equipped = {'soil': False, 'rock': False, 'imaging': False}
            for fact in static_facts:
                if fnmatch(fact, f'(equipped_for_soil_analysis {rover})'):
                    equipped['soil'] = True
                elif fnmatch(fact, f'(equipped_for_rock_analysis {rover})'):
                    equipped['rock'] = True
                elif fnmatch(fact, f'(equipped_for_imaging {rover})'):
                    equipped['imaging'] = True
            equipment[rover] = equipped
        return equipment

    def _count_remaining_data(self, state, data_type):
        """Count how many data points of a specific type are not yet communicated."""
        count = 0
        for fact in state:
            if fnmatch(fact, f'({data_type} ?w)'):
                w = fact[len(f'({data_type} '):-1].split()[0]
                count += 1
        return count

    def _is_calibrated(self, state):
        """Check if any camera is calibrated."""
        for fact in state:
            if fnmatch(fact, '(calibrated ?c ?r)'):
                return True
        return False
