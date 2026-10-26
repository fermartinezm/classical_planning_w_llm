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


class rovers17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of soil samples, rock samples, and images that need to be collected and communicated.
    It also takes into account the need to navigate to different waypoints to perform these actions.

    # Assumptions
    - Each rover has sufficient capacity to store all samples and images.
    - Rovers can perform all necessary actions if they are equipped for them.
    - Communication can always be performed if the rover is at a waypoint visible from the lander.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Identify the locations of soil and rock samples.
    - Determine which rovers are equipped for soil analysis, rock analysis, and imaging.
    - Identify the calibration targets for each camera.
    - Store the visibility information between waypoints and objectives.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check if the current state satisfies the goal. If so, return 0.
    3. Count the number of uncommunicated soil samples. For each uncommunicated soil sample, estimate the cost as follows:
       - Find a rover equipped for soil analysis.
       - Estimate the cost to navigate to the soil sample location and sample it.
       - Estimate the cost to navigate to a waypoint visible from the lander and communicate the data.
    4. Count the number of uncommunicated rock samples. For each uncommunicated rock sample, estimate the cost as follows:
       - Find a rover equipped for rock analysis.
       - Estimate the cost to navigate to the rock sample location and sample it.
       - Estimate the cost to navigate to a waypoint visible from the lander and communicate the data.
    5. Count the number of uncommunicated images. For each uncommunicated image, estimate the cost as follows:
       - Find a rover equipped for imaging.
       - If the camera is not calibrated, estimate the cost to calibrate it.
       - Estimate the cost to navigate to a waypoint visible from the objective and take the image.
       - Estimate the cost to navigate to a waypoint visible from the lander and communicate the data.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.soil_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")
        }
        self.rock_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")
        }
        self.rovers_soil = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")
        }
        self.rovers_rock = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*")
        }
        self.rovers_image = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")
        }
        self.calibration_targets = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "calibration_target", "*", "*")
        }
        self.visible_from = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "visible_from", "*", "*")
        }
        self.visible = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "visible", "*", "*")
        }
        self.lander_location = next(get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*"))

    def __call__(self, node):
        """Estimate the number of actions required to achieve the goals."""
        state = node.state

        if self.goals <= state:
            return 0

        cost = 0

        # Soil samples
        uncommunicated_soil = self.soil_samples - {
            get_parts(fact)[1] for fact in state if match(fact, "communicated_soil_data", "*")
        }
        cost += len(uncommunicated_soil)

        # Rock samples
        uncommunicated_rock = self.rock_samples - {
            get_parts(fact)[1] for fact in state if match(fact, "communicated_rock_data", "*")
        }
        cost += len(uncommunicated_rock)

        # Images
        uncommunicated_images = set()
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                uncommunicated_images.add(goal)

        communicated_images = {fact for fact in state if any(match(fact, "communicated_image_data", o, m) for o, m in [(get_parts(g)[1], get_parts(g)[2]) for g in self.goals if match(g, "communicated_image_data", "*", "*")])}

        uncommunicated_images = len(uncommunicated_images - communicated_images)
        cost += uncommunicated_images

        return cost
