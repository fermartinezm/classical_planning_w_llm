from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


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


class rovers18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of communicated data items (soil, rock, and image) that are yet to be achieved,
    and estimates the cost based on the need to navigate, sample, calibrate, take images, and communicate.

    # Assumptions
    - Each rover can perform actions independently.
    - The heuristic focuses on achieving the goals related to communicated data.
    - It assumes that the rover needs to be at a waypoint visible from the objective/lander to perform actions.
    - It assumes that the rover needs to be equipped with the necessary equipment to perform actions.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Extract static information such as visibility between waypoints, visibility from objectives,
      equipped rovers, store information, calibration targets, and on-board cameras.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check for each goal predicate if it is already achieved in the current state. If so, skip it.
    3. If the goal is to communicate soil data:
       - Check if the soil analysis is already done. If not, estimate the cost to sample soil.
       - Estimate the cost to navigate to a waypoint visible from the lander.
       - Add the cost to communicate the soil data.
    4. If the goal is to communicate rock data:
       - Check if the rock analysis is already done. If not, estimate the cost to sample rock.
       - Estimate the cost to navigate to a waypoint visible from the lander.
       - Add the cost to communicate the rock data.
    5. If the goal is to communicate image data:
       - Check if the image is already taken. If not, estimate the cost to calibrate the camera (if needed).
       - Estimate the cost to navigate to a waypoint visible from the objective.
       - Add the cost to take the image.
       - Estimate the cost to navigate to a waypoint visible from the lander.
       - Add the cost to communicate the image data.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal predicates and static information."""
        self.goals = task.goals
        static_facts = task.static

        self.visible = {}
        self.can_traverse = {}
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()
        self.store_of = {}
        self.calibration_target = {}
        self.on_board = {}
        self.supports = {}
        self.visible_from = {}
        self.at_lander = None

        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1, waypoint2 = get_parts(fact)[1], get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = set()
                self.visible[waypoint1].add(waypoint2)
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, waypoint1, waypoint2 = get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                if waypoint1 not in self.can_traverse[rover]:
                    self.can_traverse[rover][waypoint1] = set()
                self.can_traverse[rover][waypoint1].add(waypoint2)
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil_analysis.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock_analysis.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(get_parts(fact)[1])
            elif match(fact, "store_of", "*", "*"):
                store, rover = get_parts(fact)[1], get_parts(fact)[2]
                self.store_of[rover] = store
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_parts(fact)[1], get_parts(fact)[2]
                self.calibration_target[camera] = objective
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1], get_parts(fact)[2]
                if rover not in self.on_board:
                    self.on_board[rover] = set()
                self.on_board[rover].add(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1], get_parts(fact)[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1], get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            elif match(fact, "at_lander", "*", "*"):
                self.at_lander = get_parts(fact)[3]

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        soil_samples = set()
        rock_samples = set()
        images = set()
        rover_locations = {}
        stores = {}

        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                rover, waypoint = get_parts(fact)[1], get_parts(fact)[2]
                soil_samples.add(waypoint)
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, waypoint = get_parts(fact)[1], get_parts(fact)[2]
                rock_samples.add(waypoint)
            elif match(fact, "have_image", "*", "*", "*"):
                rover, objective, mode = get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]
                images.add((objective, mode))
            elif match(fact, "at", "*", "*"):
                rover, waypoint = get_parts(fact)[1], get_parts(fact)[2]
                rover_locations[rover] = waypoint
            elif match(fact, "store_of", "*", "*"):
                store, rover = get_parts(fact)[1], get_parts(fact)[2]
                stores[rover] = store

        for goal in self.goals:
            if goal in state:
                continue

            if match(goal, "communicated_soil_data", "*"):
                waypoint = get_parts(goal)[1]
                rover = next((r for r in rover_locations if self.store_of.get(r) and r in self.equipped_for_soil_analysis), None)
                if not rover:
                    heuristic_value += 100 # big penalty if no rover can do this

                if waypoint not in soil_samples:
                    heuristic_value += 2  # sampling cost

                heuristic_value += 2  # navigate and communicate

            elif match(goal, "communicated_rock_data", "*"):
                waypoint = get_parts(goal)[1]
                rover = next((r for r in rover_locations if self.store_of.get(r) and r in self.equipped_for_rock_analysis), None)
                if not rover:
                    heuristic_value += 100 # big penalty if no rover can do this

                if waypoint not in rock_samples:
                    heuristic_value += 2  # sampling cost

                heuristic_value += 2  # navigate and communicate

            elif match(goal, "communicated_image_data", "*", "*"):
                objective, mode = get_parts(goal)[1], get_parts(goal)[2]
                rover = next((r for r in rover_locations if r in self.equipped_for_imaging), None)
                if not rover:
                    heuristic_value += 100 # big penalty if no rover can do this

                if (objective, mode) not in images:
                    camera = next((c for c in self.on_board.get(rover, []) if self.supports.get(c) and mode in self.supports[c]), None)
                    if not camera:
                        heuristic_value += 100 # big penalty if no camera can do this
                    if not ("calibrated", camera, rover) in state:
                        heuristic_value += 1  # calibration cost
                    heuristic_value += 3  # navigate, take image, navigate, communicate

        return heuristic_value
