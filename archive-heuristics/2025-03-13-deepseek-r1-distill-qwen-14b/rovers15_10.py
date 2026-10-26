from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state by considering:
    - Navigation between waypoints.
    - Sample collection (soil and rock).
    - Camera calibration and imaging.
    - Data communication.

    # Assumptions:
    - Each rover can move between visible waypoints.
    - Sample collection requires being at the correct waypoint.
    - Imaging requires calibration and visible targets.
    - Data communication requires visibility between the rover and lander.

    # Heuristic Initialization
    - Extracts static facts about waypoints, visibility, and traversal.
    - Maps each waypoint to its visible neighbors and connected waypoints.

    # Step-by-Step Thinking for Computing Heuristic
    1. Extract goal requirements (soil, rock, image data).
    2. For each goal, check if it's already achieved.
    3. For unmet goals:
       a. Determine the shortest path to the required waypoint.
       b. Calculate actions for sample collection or imaging setup.
       c. Estimate communication actions.
    4. Sum the actions for all goals to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static domain information."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information about waypoints and their visibility
        self.waypoint_visibility = {}
        for fact in static_facts:
            if fact.startswith('(visible'):
                parts = fact[1:-1].split()
                if parts[1] == 'waypoint':
                    waypoint = parts[2]
                    visible_to = parts[3]
                    if waypoint not in self.waypoint_visibility:
                        self.waypoint_visibility[waypoint] = []
                    self.waypoint_visibility[waypoint].append(visible_to)

        # Extract can_traverse information for navigation
        self.can_traverse = {}
        for fact in static_facts:
            if fact.startswith('(can_traverse'):
                parts = fact[1:-1].split()
                rover = parts[1]
                from_w = parts[2]
                to_w = parts[3]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                self.can_traverse[rover][from_w] = self.can_traverse[rover].get(from_w, [])
                self.can_traverse[rover][from_w].append(to_w)

        # Extract calibration targets for each camera
        self.calibration_targets = {}
        for fact in static_facts:
            if fact.startswith('(calibration_target'):
                parts = fact[1:-1].split()
                camera = parts[1]
                obj = parts[2]
                self.calibration_targets[camera] = obj

        # Extract which cameras are on which rovers
        self.cameras_on_rovers = {}
        for fact in static_facts:
            if fact.startswith('(on_board'):
                parts = fact[1:-1].split()
                camera = parts[1]
                rover = parts[2]
                if rover not in self.cameras_on_rovers:
                    self.cameras_on_rovers[rover] = []
                self.cameras_on_rovers[rover].append(camera)

        # Extract which modes each camera supports
        self.camera_modes = {}
        for fact in static_facts:
            if fact.startswith('(supports'):
                parts = fact[1:-1].split()
                camera = parts[1]
                mode = parts[2]
                if camera not in self.camera_modes:
                    self.camera_modes[camera] = []
                self.camera_modes[camera].append(mode)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        state_facts = set(state)

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        total_actions = 0

        # Extract goal requirements
        soil_goals = set()
        rock_goals = set()
        image_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                soil_goals.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                rock_goals.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                if (obj, mode) not in image_goals:
                    image_goals[(obj, mode)] = 0
                image_goals[(obj, mode)] += 1

        # Check if all goals are already met
        if not soil_goals and not rock_goals and not image_goals:
            return 0

        # Process each type of goal
        # 1. Soil and Rock Analysis
        for goal_w in soil_goals | rock_goals:
            if goal_w in soil_goals:
                # Soil analysis
                has_soil = any(match(f, 'have_soil_analysis', '*', goal_w) for f in state_facts)
                if not has_soil:
                    # Find the rover equipped for soil analysis
                    rover = None
                    for fact in state_facts:
                        if match(fact, 'equipped_for_soil_analysis', '*'):
                            rover = re.findall(r'\w+', fact)[1]
                            break
                    if rover:
                        # Find the shortest path from current position to goal_w
                        path_length = 1  # Simplified as all moves are 1 step
                        total_actions += path_length + 1  # Move and sample

                        # Check if communication is needed
                        if not any(match(f, 'communicated_soil_data', goal_w) for f in state_facts):
                            # Find the lander's position
                            lander_pos = None
                            for fact in state_facts:
                                if match(fact, 'at_lander', '*', '*'):
                                    lander_pos = re.findall(r'\w+', fact)[2]
                                    break
                            if lander_pos:
                                # Move to a waypoint visible to lander
                                visible_waypoints = self.waypoint_visibility.get(rover, {}).get(rover, [])
                                if visible_waypoints:
                                    total_actions += 1  # Move to communication point
                                    total_actions += 1  # Communicate
            if goal_w in rock_goals:
                # Similar logic for rock analysis
                has_rock = any(match(f, 'have_rock_analysis', '*', goal_w) for f in state_facts)
                if not has_rock:
                    # Find the rover equipped for rock analysis
                    rover = None
                    for fact in state_facts:
                        if match(fact, 'equipped_for_rock_analysis', '*'):
                            rover = re.findall(r'\w+', fact)[1]
                            break
                    if rover:
                        # Find the shortest path
                        path_length = 1
                        total_actions += path_length + 1

                        # Check communication
                        if not any(match(f, 'communicated_rock_data', goal_w) for f in state_facts):
                            lander_pos = None
                            for fact in state_facts:
                                if match(fact, 'at_lander', '*', '*'):
                                    lander_pos = re.findall(r'\w+', fact)[2]
                                    break
                            if lander_pos:
                                visible_waypoints = self.waypoint_visibility.get(rover, {}).get(rover, [])
                                if visible_waypoints:
                                    total_actions += 1  # Move
                                    total_actions += 1  # Communicate

        # 2. Image Analysis
        for (obj, mode), count in image_goals.items():
            if count == 0:
                continue
            # Check if image is already communicated
            if any(match(f, 'communicated_image_data', obj, mode) for f in state_facts):
                continue

            # Find a rover equipped for imaging
            rover = None
            for fact in state_facts:
                if match(fact, 'equipped_for_imaging', '*'):
                    rover = re.findall(r'\w+', fact)[1]
                    break
            if not rover:
                continue

            # Find a camera on the rover that supports the mode
            camera = None
            for cam in self.cameras_on_rovers.get(rover, []):
                if mode in self.camera_modes.get(cam, []):
                    camera = cam
                    break
            if not camera:
                continue

            # Check if camera is calibrated
            calibrated = any(match(f, 'calibrated', camera, rover) for f in state_facts)
            if not calibrated:
                # Find the calibration target
                target = self.calibration_targets.get(camera, None)
                if target:
                    # Find the waypoint where the target is visible
                    visible_waypoints = []
                    for fact in state_facts:
                        if match(fact, 'visible_from', target, '*'):
                            waypoint = re.findall(r'\w+', fact)[2]
                            visible_waypoints.append(waypoint)
                    if visible_waypoints:
                        # Move to the waypoint
                        total_actions += 1  # Move
                        # Calibrate
                        total_actions += 1

            # Take the image
            total_actions += 1  # Take image

            # Communicate the image
            # Find the lander's position
            lander_pos = None
            for fact in state_facts:
                if match(fact, 'at_lander', '*', '*'):
                    lander_pos = re.findall(r'\w+', fact)[2]
                    break
            if lander_pos:
                # Find a waypoint visible to the lander where the rover can communicate
                visible_waypoints = self.waypoint_visibility.get(rover, {}).get(rover, [])
                if visible_waypoints:
                    total_actions += 1  # Move to communication point
                    total_actions += 1  # Communicate

        return total_actions
