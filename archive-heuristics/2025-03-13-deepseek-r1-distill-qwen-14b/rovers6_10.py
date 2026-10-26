from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to communicate all required data points (soil, rock, and images) from their respective locations.

    # Assumptions:
    - The rover may need to move between waypoints to collect and communicate data.
    - Each data type (soil, rock, image) may require specific actions to be collected and communicated.
    - The rover can carry multiple samples, but each communication action is counted separately.

    # Heuristic Initialization
    - Extract goal conditions for each data type (soil, rock, image) and their respective locations.
    - Extract static facts including can_traverse, visible, and calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of soil, rock, and image data points that still need to be communicated.
    2. For each data type, determine if the rover has the necessary information and hasn't communicated it yet.
    3. For images, check if the camera is calibrated and if the objective is visible.
    4. Calculate the cost based on the number of remaining communications and the steps required to move to the necessary locations.
    5. Sum the costs for all data types to get the total estimated actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.waypoint_visibility = {}
        self.can_traverse = {}
        self.calibration_targets = {}
        self.camera_support = {}
        self.image_objectives = {}

        # Populate waypoint visibility
        for fact in self.static:
            if fact.startswith('(visible'):
                parts = fact[1:-1].split()
                if len(parts) == 3:
                    waypoint, visible_from, _ = parts
                    if waypoint not in self.waypoint_visibility:
                        self.waypoint_visibility[waypoint] = []
                    self.waypoint_visibility[waypoint].append(visible_from)
                elif len(parts) == 4:
                    visible_from, waypoint, _ = parts[1:]
                    if waypoint not in self.waypoint_visibility:
                        self.waypoint_visibility[waypoint] = []
                    self.waypoint_visibility[waypoint].append(visible_from)

        # Populate can_traverse information
        for fact in self.static:
            if fact.startswith('(can_traverse'):
                parts = fact[1:-1].split()
                rover, from_wpt, to_wpt = parts
                key = (rover, from_wpt)
                if key not in self.can_traverse:
                    self.can_traverse[key] = []
                self.can_traverse[key].append(to_wpt)

        # Populate calibration targets
        for fact in self.static:
            if fact.startswith('(calibration_target'):
                parts = fact[1:-1].split()
                camera, objective = parts
                self.calibration_targets[rover] = (camera, objective)

    def __call__(self, node):
        """Estimate the minimum cost to achieve the goal state."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        cost = 0

        # Count remaining soil data to communicate
        soil_communicated = 0
        for goal in self.goals:
            if match(goal, 'communicated_soil_data', '*'):
                soil_communicated += 1
        soil_remaining = sum(1 for fact in state if match(fact, 'have_soil_analysis', '*', '*')) - soil_communicated
        cost += soil_remaining * 2  # Communicate each soil sample

        # Count remaining rock data to communicate
        rock_communicated = 0
        for goal in self.goals:
            if match(goal, 'communicated_rock_data', '*'):
                rock_communicated += 1
        rock_remaining = sum(1 for fact in state if match(fact, 'have_rock_analysis', '*', '*')) - rock_communicated
        cost += rock_remaining * 2  # Communicate each rock sample

        # Count remaining image data to communicate
        image_communicated = 0
        for goal in self.goals:
            if match(goal, 'communicated_image_data', '*', '*', '*'):
                image_communicated += 1
        image_goals = sum(1 for fact in self.goals if match(fact, 'communicated_image_data', '*', '*', '*'))
        image_remaining = image_goals - image_communicated
        cost += image_remaining * 3  # Each image requires calibration and communication

        # Check if images need calibration
        for fact in state:
            if match(fact, 'calibrated', '*', '*'):
                pass
            else:
                # Assume calibration is needed if not already done
                cost += 1  # Calibration action

        return cost
