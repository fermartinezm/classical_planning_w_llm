from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to communicate all required soil, rock, and image data.

    # Assumptions:
    - Each sample (soil and rock) must be collected and communicated.
    - Images must be taken and communicated for each objective and mode.
    - The rover can move between waypoints if they are traversable and visible.
    - Communication requires visibility between the rover's location and the lander.

    # Heuristic Initialization
    - Extract static facts to determine visible waypoints, traversable paths, and calibration targets.
    - Identify goal locations for each sample and image.

    # Step-by-Step Thinking for Computing Heuristic
    1. Extract the set of samples (soil and rock) that need to be communicated.
    2. For each sample, find the closest waypoint that can be reached by the rover and from where communication is possible.
    3. Calculate the minimal number of moves required for the rover to reach that waypoint.
    4. Sum the steps for all necessary communications, including moving, collecting, and communicating each sample.
    5. Add steps for taking and communicating images if required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Extract visible waypoints from lander
        self.visible_from_lander = set()
        for fact in static_facts:
            if match(fact, "(visible ?w - waypoint ?y - waypoint)"):
                w, y = get_parts(fact)[1], get_parts(fact)[3]
                if match(fact, "* waypoint1"):
                    self.visible_from_lander.add(w)

        # Build can_traverse graph
        self.traversal_graph = {}
        for fact in static_facts:
            if match(fact, "(can_traverse ?r - rover ?x - waypoint ?y - waypoint)"):
                x, y = get_parts(fact)[2], get_parts(fact)[4]
                if x not in self.traversal_graph:
                    self.traversal_graph[x] = []
                self.traversal_graph[x].append(y)
                if y not in self.traversal_graph:
                    self.traversal_graph[y] = []
                self.traversal_graph[y].append(x)

        # Extract calibration targets
        self.calibration_targets = {}
        for fact in static_facts:
            if match(fact, "(calibration_target ?i - camera ?o - objective)"):
                i, o = get_parts(fact)[1], get_parts(fact)[3]
                self.calibration_targets[i] = o

        # Store goal locations for each sample
        self.goal_samples = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "communicated_soil_data" or predicate == "communicated_rock_data":
                w = args[0]
                self.goal_samples[w] = predicate

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        current_location = None
        has_soil = set()
        has_rock = set()
        communicated_soil = set()
        communicated_rock = set()
        communicated_images = set()

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                current_location = parts[2]
            elif parts[0] == "have_soil_analysis":
                has_soil.add(parts[2])
            elif parts[0] == "have_rock_analysis":
                has_rock.add(parts[2])
            elif parts[0] == "communicated_soil_data":
                communicated_soil.add(parts[1])
            elif parts[0] == "communicated_rock_data":
                communicated_rock.add(parts[1])
            elif parts[0] == "communicated_image_data":
                communicated_images.add(f"{parts[1]}_{parts[2]}")

        # Count required actions
        total_cost = 0

        # Communicate soil samples
        for w in has_soil:
            if w not in communicated_soil:
                # Find shortest path to a waypoint visible from lander
                path = self.find_shortest_path(current_location, self.visible_from_lander)
                if path:
                    total_cost += len(path) - 1  # Moves to reach the waypoint
                    total_cost += 1  # Communicate action
                else:
                    total_cost += float('inf')  # Unreachable

        # Communicate rock samples
        for w in has_rock:
            if w not in communicated_rock:
                path = self.find_shortest_path(current_location, self.visible_from_lander)
                if path:
                    total_cost += len(path) - 1
                    total_cost += 1
                else:
                    total_cost += float('inf')

        # Communicate images
        # (This part can be extended based on specific image requirements)

        return total_cost

    def find_shortest_path(self, start, targets):
        """Breadth-first search to find the shortest path to any target waypoint."""
        visited = set()
        queue = deque()
        queue.append((start, []))

        while queue:
            current, path = queue.popleft()
            if current in targets and current != start:
                return path + [current]
            if current in visited:
                continue
            visited.add(current)
            if current in self.traversal_graph:
                for neighbor in self.traversal_graph[current]:
                    if neighbor not in visited:
                        new_path = path + [current]
                        queue.append((neighbor, new_path))
        return None

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ?x - rover ?y - waypoint)".
    - `args`: The expected pattern (wildcards `?` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))
