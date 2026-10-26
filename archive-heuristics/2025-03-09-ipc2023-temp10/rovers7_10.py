from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package5 city3-2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class rovers7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of communicated data goals (soil, rock, and image) that are not yet achieved and estimates the cost based on the following:
    - Navigating to locations with samples or objectives.
    - Sampling soil and rock samples.
    - Taking images.
    - Communicating data to the lander.

    # Assumptions
    - Each uncommunicated data requires at least one navigate action to reach the sample/objective location.
    - Each sample/image requires one sample/take_image action.
    - Each communicated data requires one communicate action.
    - The heuristic does not account for the rover's capabilities or the need for calibration.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Identify the soil samples, rock samples, and objectives from the initial state and static facts.
    - Identify the lander location from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state.
    2. Count the number of uncommunicated soil, rock, and image data goals.
    3. For each uncommunicated data goal, add an estimated cost:
        - 1 for navigating to the location of the sample/objective.
        - 1 for sampling/taking the image.
        - 1 for communicating the data.
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.soil_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")
        }
        self.rock_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")
        }
        self.objectives = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "visible_from", "*", "*")
        }
        self.lander_location = next(
            (get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None
        )

    def __call__(self, node):
        """
        Estimate the number of actions required to achieve the goals.
        """
        state = node.state
        uncommunicated_soil_data = sum(
            1 for w in self.soil_samples if f"(communicated_soil_data {w})" not in state
        )
        uncommunicated_rock_data = sum(
            1 for w in self.rock_samples if f"(communicated_rock_data {w})" not in state
        )
        uncommunicated_image_data = sum(
            1
            for o in self.objectives
            for m in ["colour", "high_res", "low_res"]
            if f"(communicated_image_data {o} {m})" not in state
        )

        # Estimate cost: navigate + sample/image + communicate
        total_cost = (
            uncommunicated_soil_data * 3
            + uncommunicated_rock_data * 3
            + uncommunicated_image_data * 3
        )

        return total_cost
