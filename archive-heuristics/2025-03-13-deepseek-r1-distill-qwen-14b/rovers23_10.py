from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to collect all required samples and communicate their data.

    # Assumptions:
    - The rover must collect soil and rock samples from specified waypoints.
    - The rover must communicate the collected data to the lander.
    - Communication can only occur if the rover and lander are visible to each other.

    # Heuristic Initialization
    - Extract goal conditions to identify which data points need to be communicated.
    - Parse static facts to determine connectivity between waypoints and calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all waypoints that require soil and rock samples to be collected and communicated.
    2. For each required sample:
       a. Check if the sample has already been collected and communicated.
       b. If not, calculate the number of actions needed to navigate to the waypoint, collect the sample, and communicate it.
    3. Sum the actions needed for all required samples to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.waypoint_connections = self._extract_waypoint_connections()
        self.communication_links = self._extract_communication_links()
        self.calibration_targets = self._extract_calibration_targets()

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        if self._goal_reached(state):
            return 0

        total_actions = 0

        # Check for each required data point
        for fact in self.goals:
            predicate, *args = self._parse_fact(fact)
            if predicate == "communicated_image_data":
                objective, mode = args
                total_actions += self._estimate_image_communication(objective, mode, state)
            elif predicate in ["communicated_soil_data", "communicated_rock_data"]:
                waypoint = args[0]
                total_actions += self._estimate_sample_communication(predicate, waypoint, state)

        return total_actions

    def _parse_fact(self, fact):
        """Extract components of a PDDL fact."""
        return fact[1:-1].split()

    def _goal_reached(self, state):
        """Check if all goal conditions are met."""
        return all(fact in state for fact in self.goals)

    def _extract_waypoint_connections(self):
        """Extract all connected waypoints from static facts."""
        connections = set()
        for fact in self.static:
            if self._match(fact, "can_traverse", "*", "*", "*"):
                _, x, y, z = self._parse_fact(fact)
                connections.add((x, y, z))
                connections.add((z, y, x))
        return connections

    def _extract_communication_links(self):
        """Extract communication visibility between locations."""
        links = set()
        for fact in self.static:
            if self._match(fact, "visible", "*", "*"):
                _, x, y = self._parse_fact(fact)
                links.add((x, y))
                links.add((y, x))
        return links

    def _extract_calibration_targets(self):
        """Extract calibration targets for each camera."""
        targets = {}
        for fact in self.static:
            if self._match(fact, "calibration_target", "*", "*"):
                _, camera, target = self._parse_fact(fact)
                if camera not in targets:
                    targets[camera] = []
                targets[camera].append(target)
        return targets

    def _match(self, fact, *args):
        """Check if a fact matches a given pattern."""
        parts = self._parse_fact(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def _estimate_image_communication(self, objective, mode, state):
        """Estimate actions needed to communicate an image."""
        # Check if the image is already communicated
        image_fact = f"(have_image ?r - rover {objective} {mode})"
        if any(self._match(f, image_fact) for f in state):
            return 0

        # Find a rover that can take images and is at a waypoint visible to the objective
        rover = None
        for fact in state:
            if self._match(fact, "at", "*", "*"):
                rover, waypoint = self._parse_fact(fact)[1:]
                break

        # Find a communication path from current waypoint to lander
        # This is a simplified estimate; in a full implementation, BFS could find the shortest path
        return 4  # navigate to lander, communicate, return (example estimate)

    def _estimate_sample_communication(self, predicate, waypoint, state):
        """Estimate actions needed to communicate a sample."""
        # Check if the sample has been communicated
        communicated = f"(communicated_{predicate[11:-1]}_data {waypoint})"
        if communicated in state:
            return 0

        # Check if the sample has been collected
        has_sample = f"(have_soil_analysis" if predicate.startswith("communicated_soil") else "(have_rock_analysis"
        has_sample += f" ?r - rover {waypoint})"

        if not any(self._match(f, has_sample) for f in state):
            # Need to collect the sample
            return 5  # navigate, collect, communicate (example estimate)

        # Sample is collected but not communicated
        return 3  # navigate to lander, communicate, return (example estimate)
