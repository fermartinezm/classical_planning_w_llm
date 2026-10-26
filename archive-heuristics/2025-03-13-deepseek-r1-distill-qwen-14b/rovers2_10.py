<code-file-heuristic-3>
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to:
    1. Collect all required soil and rock samples
    2. Take and communicate all required images
    3. Communicate all collected data to the lander

    # Assumptions:
    - Each sample (soil/rock) requires one action to collect and one to communicate
    - Each image requires calibration, capture, and communication
    - The rover can carry one sample at a time
    - Navigation between visible waypoints costs one action per move
    - If multiple samples are needed from the same waypoint, only one trip is required

    # Heuristic Initialization
    - Extracts goal conditions (which data needs to be communicated)
    - Maps waypoints to their sample types (soil/rock) using static facts
    - Maps objectives to visible waypoints using static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. Check which soil and rock samples need to be communicated
    2. For each missing sample:
       a. Check if the sample has been collected
       b. If not, estimate actions needed to collect it
    3. Check which images need to be communicated
    4. For each missing image:
       a. Check if the image has been taken
       b. If not, estimate actions needed to take it
    5. Sum all required actions for missing data points
    """

    def __init__(self, task):
        """Initialize the heuristic with task-specific information."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract static information into useful data structures
        self.waypoint_samples = {}
        for fact in static_facts:
            if fact.startswith("(at_soil_sample ") or fact.startswith("(at_rock_sample "):
                parts = fact[1:-1].split()
                waypoint = parts[1]
                sample_type = parts[0].split('_')[1]  # 'soil' or 'rock'
                if waypoint not in self.waypoint_samples:
                    self.waypoint_samples[waypoint] = []
                self.waypoint_samples[waypoint].append(sample_type)

        self.objective_views = {}
        for fact in static_facts:
            if fact.startswith("(visible_from "):
                parts = fact[1:-1].split()
                obj = parts[1]
                waypoint = parts[2]
                if obj not in self.objective_views:
                    self.objective_views[obj] = []
                self.objective_views[obj].append(waypoint)

    def __call__(self, node):
        """Estimate the minimum cost to achieve all goal conditions."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        heuristic_cost = 0

        # Check soil communication goals
        soil_communicated = set()
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                soil_communicated.add(goal.split()[1])
        for waypoint in self.waypoint_samples:
            if 'soil' in self.waypoint_samples[waypoint]:
                if waypoint not in soil_communicated:
                    # Need to collect and communicate soil sample
                    heuristic_cost += 2  # 1 to collect, 1 to communicate

        # Check rock communication goals
        rock_communicated = set()
        for goal in self.goals:
            if match(goal, "communicated_rock_data", "*"):
                rock_communicated.add(goal.split()[1])
        for waypoint in self.waypoint_samples:
            if 'rock' in self.waypoint_samples[waypoint]:
                if waypoint not in rock_communicated:
                    # Need to collect and communicate rock sample
                    heuristic_cost += 2  # 1 to collect, 1 to communicate

        # Check image communication goals
        image_communicated = set()
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                obj = goal.split()[1]
                mode = goal.split()[2]
                key = (obj, mode)
                image_communicated.add(key)
        for obj in self.objective_views:
            for mode in ['high_res', 'low_res', 'colour']:
                if (obj, mode) not in image_communicated:
                    # Need to take and communicate image
                    heuristic_cost += 3  # 1 calibrate, 1 capture, 1 communicate

        return heuristic_cost
</code-file-heuristic-3>