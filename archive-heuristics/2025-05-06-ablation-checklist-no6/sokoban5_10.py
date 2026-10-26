from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of shortest path distances for each box
    to its goal, plus the minimum shortest path distance for the robot
    to reach any location adjacent to any misplaced box.

    Distances are computed on the grid graph defined by 'adjacent' facts,
    ignoring dynamic obstacles (other boxes, robot).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        precomputing distances.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Build the adjacency list graph from static facts
        self.adjacencies = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                locations.add(loc1)
                locations.add(loc2)
                # Treat as undirected graph for distance calculation
                self.adjacencies.setdefault(loc1, []).append(loc2)
                self.adjacencies.setdefault(loc2, []).append(loc1) # Add reverse edge

        self.locations = list(locations) # Store list of all locations

        # Precompute all-pairs shortest paths using BFS from each location
        self.distances = {}
        for start_loc in self.locations:
            # Run BFS from start_loc to all other locations
            queue = [(start_loc, 0)]
            visited = {start_loc: 0} # Store distance in visited set

            head = 0 # Use index for efficient queue simulation with list
            while head < len(queue):
                current_loc, dist = queue[head]
                head += 1

                for neighbor in self.adjacencies.get(current_loc, []):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

            # Store distances from start_loc
            for end_loc in self.locations:
                 self.distances[(start_loc, end_loc)] = visited.get(end_loc, float('inf'))


        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location
            # Assuming goals are only (at box location) and involve 'at' predicate

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        
        # If robot location is not found, the state is likely invalid or terminal unsolvable
        if robot_loc is None:
             return float('inf')

        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                # Check if the object is one of the boxes required by the goal
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        total_box_distance = 0
        min_robot_distance_to_any_box_adjacent = float('inf')
        misplaced_boxes_count = 0

        # Iterate over the boxes that have a goal location
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            # If a box required by the goal is not found in the state, it's an issue.
            # Assuming all goal boxes are always present in the state.
            if current_location is None:
                 # This should not happen in valid Sokoban states, but handle defensively.
                 # A box required by the goal is missing from the state.
                 return float('inf')

            if current_location != goal_location:
                misplaced_boxes_count += 1
                
                # Distance for the box to its goal on the empty grid
                dist_box_goal = self.distances.get((current_location, goal_location), float('inf'))
                
                # If any box cannot reach its goal location on the empty grid, it's likely unsolvable
                # without complex maneuvers (like moving other boxes out of the way and back),
                # which this simple heuristic doesn't model. Treat as very high cost.
                if dist_box_goal == float('inf'):
                    return float('inf') # Cannot reach goal even on empty grid

                total_box_distance += dist_box_goal

                # Minimum distance for the robot to reach any location adjacent to this box
                min_dist_robot_to_this_box_adjacent = float('inf')
                # Check if current_location is in adjacencies keys (it should be if it's a valid location)
                if current_location in self.adjacencies:
                    for adj_loc in self.adjacencies[current_location]:
                        dist_robot_to_adj = self.distances.get((robot_loc, adj_loc), float('inf'))
                        min_dist_robot_to_this_box_adjacent = min(min_dist_robot_to_this_box_adjacent, dist_robot_to_adj)
                # else: The box is at a location not in the graph? Should not happen. Treat as unreachable.
                #     min_dist_robot_to_this_box_adjacent = float('inf') # Already initialized to inf

                # Update the minimum robot distance needed for *any* misplaced box
                min_robot_distance_to_any_box_adjacent = min(min_robot_distance_to_any_box_adjacent, min_dist_robot_to_this_box_adjacent)

        # If no boxes are misplaced, total_box_distance is 0 and misplaced_boxes_count is 0.
        # The initial goal check handles this case (returns 0).

        # If there are misplaced boxes but the robot cannot reach any adjacent location
        # of any of them, it's likely stuck.
        if misplaced_boxes_count > 0 and min_robot_distance_to_any_box_adjacent == float('inf'):
             return float('inf')

        # The heuristic is the sum of box distances plus the cost for the robot
        # to get into position for *at least one* of the pushes.
        # This is a simple combination that captures the two main costs:
        # moving the boxes and getting the robot to a box.
        return total_box_distance + min_robot_distance_to_any_box_adjacent
