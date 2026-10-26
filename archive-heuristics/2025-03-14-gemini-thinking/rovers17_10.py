from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

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

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal state in the Rovers domain.
    It considers the necessary steps for each goal, such as navigating to locations,
    sampling soil or rock, calibrating cameras, taking images, and communicating data.
    The heuristic focuses on the most costly actions and their dependencies to provide a reasonable estimate.

    # Assumptions:
    - The heuristic assumes that for each goal, the rover will need to perform a sequence of actions.
    - It prioritizes achieving each goal independently and takes the maximum cost among all goals.
    - It assumes that rovers will need to navigate to locations to perform actions.
    - It does not explicitly consider resource constraints like store capacity, assuming it is always managed appropriately by the planner.

    # Heuristic Initialization
    - Extracts static information from the task, such as:
        - Visibility relations between waypoints and from objectives.
        - Traverse capabilities of rovers.
        - Equipment of rovers (soil, rock, imaging).
        - Camera details (on-board, supported modes, calibration targets).
        - Locations of soil and rock samples.
        - Lander location.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If not achieved, determine the type of goal (communicate soil, rock, or image data).
    3. Estimate the minimum actions required to achieve this goal:
        - For `communicated_soil_data(waypoint)`:
            - If `communicated_soil_data(waypoint)` is not true:
                - Check if `have_soil_analysis(rover, waypoint)` is true. If not:
                    - Need to `sample_soil` at `waypoint`. This requires:
                        - Rover to be at `waypoint`. (Navigation cost if not there)
                        - Rover to be equipped for soil analysis. (Check capability)
                        - Rover's store to be empty. (Potentially `drop` action if full)
                - Need to `communicate_soil_data`. This requires:
                    - Rover to be at a waypoint visible to the lander. (Navigation cost if not there)
        - For `communicated_rock_data(waypoint)`:
            - Similar logic as `communicated_soil_data`, but for rock samples and `sample_rock` action.
        - For `communicated_image_data(objective, mode)`:
            - If `communicated_image_data(objective, mode)` is not true:
                - Check if `have_image(rover, objective, mode)` is true. If not:
                    - Need to `take_image` of `objective` in `mode`. This requires:
                        - Rover to be at a waypoint visible from `objective`. (Navigation cost if not there)
                        - Camera on board the rover that supports `mode`. (Check camera capabilities)
                        - Camera to be calibrated for `objective`. (Calibration cost if not calibrated)
                            - `calibrate` action requires:
                                - Rover to be at a waypoint visible from `objective`. (Navigation cost if not there)
                                - Rover to be equipped for imaging. (Check capability)
                                - Camera to be on board the rover and target `objective`. (Check camera and target)
                - Need to `communicate_image_data`. This requires:
                    - Rover to be at a waypoint visible to the lander. (Navigation cost if not there)

    4. Sum up the estimated costs for each goal predicate. Since we want a heuristic that is not necessarily admissible but efficient, we can take the maximum of the estimated costs for each goal predicate instead of summing them. This focuses on the most difficult goal to achieve.

    In this simplified heuristic, we will count 1 action for each major step: navigate, sample/calibrate/take_image, communicate, drop.
    We will estimate the cost for each goal and return the maximum cost among all goals.
    This is a simplification and can be improved by considering shortest path navigation, but it serves as a reasonable starting point for a domain-dependent heuristic.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.at_lander_location = next((get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None)
        self.can_traverse = {} # rover -> {waypoint -> [waypoint]}
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()
        self.store_of = {} # store -> rover
        self.calibration_target = {} # camera -> objective
        self.on_board = {} # camera -> rover
        self.supports = {} # camera -> [mode]
        self.visible = {} # waypoint -> [waypoint]
        self.visible_from = {} # objective -> [waypoint]
        self.at_soil_sample = set()
        self.at_rock_sample = set()

        for fact in static_facts:
            if match(fact, "can_traverse", "?r", "?x", "?y"):
                rover = get_parts(fact)[1]
                waypoint_x = get_parts(fact)[2]
                waypoint_y = get_parts(fact)[3]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                if waypoint_x not in self.can_traverse[rover]:
                    self.can_traverse[rover][waypoint_x] = []
                self.can_traverse[rover][waypoint_x].append(waypoint_y)
            elif match(fact, "equipped_for_soil_analysis", "?r"):
                self.equipped_for_soil_analysis.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "?r"):
                self.equipped_for_rock_analysis.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "?r"):
                self.equipped_for_imaging.add(get_parts(fact)[1])
            elif match(fact, "store_of", "?s", "?r"):
                self.store_of[get_parts(fact)[1]] = get_parts(fact)[2]
            elif match(fact, "calibration_target", "?c", "?o"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                if camera not in self.calibration_target:
                    self.calibration_target[camera] = []
                self.calibration_target[camera].append(objective)
            elif match(fact, "on_board", "?c", "?r"):
                camera = get_parts(fact)[1]
                self.on_board[camera] = get_parts(fact)[2]
            elif match(fact, "supports", "?c", "?m"):
                camera = get_parts(fact)[1]
                mode = get_parts(fact)[2]
                if camera not in self.supports:
                    self.supports[camera] = []
                self.supports[camera].append(mode)
            elif match(fact, "visible", "?w1", "?w2"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = []
                self.visible[waypoint1].append(waypoint2)
            elif match(fact, "visible_from", "?o", "?w"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = []
                self.visible_from[objective].append(waypoint)
            elif match(fact, "at_soil_sample", "?w"):
                self.at_soil_sample.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "?w"):
                self.at_rock_sample.add(get_parts(fact)[1])

    def __call__(self, node):
        """Estimate the heuristic value for the given state."""
        state = node.state
        goal_cost = 0

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved, cost is 0 for this goal part

            current_goal_cost = 0
            goal_parts = get_parts(goal)
            goal_predicate = goal_parts[0]

            if goal_predicate == 'communicated_soil_data':
                waypoint = goal_parts[1]
                rover = next((r for r in self.equipped_for_soil_analysis if r is not None), None) # Assume at least one rover is equipped
                if rover:
                    if not any(match(fact, 'have_soil_analysis', rover, waypoint) for fact in state):
                        current_goal_cost += 1 # sample_soil
                    current_goal_cost += 1 # communicate_soil_data
                else:
                    current_goal_cost += 2 # No rover can do soil analysis, should be infinite in admissible heuristic, but here just high cost.

            elif goal_predicate == 'communicated_rock_data':
                waypoint = goal_parts[1]
                rover = next((r for r in self.equipped_for_rock_analysis if r is not None), None) # Assume at least one rover is equipped
                if rover:
                    if not any(match(fact, 'have_rock_analysis', rover, waypoint) for fact in state):
                        current_goal_cost += 1 # sample_rock
                    current_goal_cost += 1 # communicate_rock_data
                else:
                    current_goal_cost += 2 # No rover can do rock analysis

            elif goal_predicate == 'communicated_image_data':
                objective = goal_parts[1]
                mode = goal_parts[2]
                rover = next((r for r in self.equipped_for_imaging if r is not None), None) # Assume at least one rover is equipped
                if rover:
                    camera = next((c for c in self.on_board if self.on_board[c] == rover and mode in self.supports.get(c, []) and objective in self.calibration_target.get(c, [])), None)
                    if not any(match(fact, 'have_image', rover, objective, mode) for fact in state):
                        if not any(match(fact, 'calibrated', camera, rover) for fact in state if camera is not None):
                            current_goal_cost += 1 # calibrate
                        current_goal_cost += 1 # take_image
                    current_goal_cost += 1 # communicate_image_data
                else:
                    current_goal_cost += 3 # No rover can do imaging

            goal_cost = max(goal_cost, current_goal_cost) # Max heuristic

        return goal_cost
