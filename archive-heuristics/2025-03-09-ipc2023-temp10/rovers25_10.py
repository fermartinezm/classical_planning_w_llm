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


class rovers25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of communicated data goals (soil, rock, and image) that are not yet achieved and estimates the cost
    based on the need to navigate to sample locations, calibrate cameras, take images, and communicate data.

    # Assumptions
    - Each rover has the necessary equipment to perform its tasks.
    - Rovers can move directly to any waypoint visible from their current location.
    - Communicating data requires the rover to be at a waypoint visible from the lander.

    # Heuristic Initialization
    - Extract the goal predicates (communicated_soil_data, communicated_rock_data, communicated_image_data).
    - Identify the locations of soil and rock samples.
    - Identify which objectives are visible from which waypoints.
    - Identify calibration targets for each camera.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state satisfies the goal conditions. If so, the heuristic value is 0.
    2. Count the number of communicated data goals (soil, rock, image) that are not yet achieved.
    3. For each unachieved goal, estimate the cost as follows:
        - If the goal is to communicate soil data:
            - Check if the soil analysis has been performed at the waypoint. If not, estimate the cost to navigate to the soil sample location and sample the soil.
            - Estimate the cost to navigate to a waypoint visible from the lander and communicate the data.
        - If the goal is to communicate rock data:
            - Check if the rock analysis has been performed at the waypoint. If not, estimate the cost to navigate to the rock sample location and sample the rock.
            - Estimate the cost to navigate to a waypoint visible from the lander and communicate the data.
        - If the goal is to communicate image data:
            - Check if the image has been taken. If not, estimate the cost to calibrate the camera (if needed), navigate to a waypoint visible from the objective, and take the image.
            - Estimate the cost to navigate to a waypoint visible from the lander and communicate the data.
    4. Sum the estimated costs for all unachieved goals.
    5. Return the total estimated cost.
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
        self.visible_from = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)

        self.calibration_targets = {}
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_targets[camera] = objective

        self.lander_location = next(get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*"))

        self.visible_waypoints = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.visible_waypoints:
                    self.visible_waypoints[waypoint1] = set()
                self.visible_waypoints[waypoint1].add(waypoint2)

    def __call__(self, node):
        """Estimate the cost to reach the goal from the given state."""
        state = node.state

        if task.goal_reached(state):
            return 0

        unachieved_goals = self.goals - state
        total_cost = 0

        for goal in unachieved_goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                if f"(have_soil_analysis *{waypoint})" not in [f.replace(waypoint, '*{waypoint}') for f in state]:
                    total_cost += 2  # Navigate and sample soil
                total_cost += 1  # Communicate soil data

            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                if f"(have_rock_analysis *{waypoint})" not in [f.replace(waypoint, '*{waypoint}') for f in state]:
                    total_cost += 2  # Navigate and sample rock
                total_cost += 1  # Communicate rock data

            elif predicate == "communicated_image_data":
                objective = parts[1]
                mode = parts[2]
                if f"(have_image *{objective} {mode})" not in [f.replace(objective, '*{objective}').replace(mode, f'*{mode}') for f in state]:
                    # Find a camera that supports the mode and is on a rover
                    camera = next((cam for cam in self.calibration_targets if self.calibration_targets[cam] == objective), None)
                    if camera is None:
                        continue

                    if f"(calibrated {camera} *)" not in [f.replace(camera, '*{camera}') for f in state]:
                        total_cost += 1  # Calibrate camera
                    total_cost += 1  # Take image
                total_cost += 1  # Communicate image data

        return total_cost
