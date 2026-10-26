from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def bfs_distance(graph, start, end):
    """Calculates the shortest path distance between two locations in the graph."""
    if start == end:
        return 0
    # Check if start or end are valid nodes in the graph
    if start not in graph or end not in graph:
         # If either location is not in the graph (e.g., isolated wall location),
         # it's unreachable. Return a large value.
         return 1000 # A large constant indicating difficulty/impossibility

    queue = deque([(start, 0)]) # (location, distance)
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        # Ensure current_loc has neighbors defined in the graph
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    # If BFS completes without finding the end, there is no path.
    return 1000 # No path found, treat as very high cost


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distances
    for each box from its current location to its goal location.

    # Assumptions
    - The grid structure and connectivity are defined by the 'adjacent' predicates.
    - The shortest path distance between locations represents the minimum number
      of push actions required to move a box between those locations, ignoring
      robot movement and obstacles (other boxes, walls not defined by adjacent).
    - The heuristic is admissible because each push action can reduce the
      shortest path distance of the pushed box to its goal by at most 1.
      (Note: This is a simplification; a push might not always reduce the distance
       or might require multiple robot moves first. However, for a simple
       admissible heuristic, this is a common approach).
    - The heuristic is 0 if and only if all boxes are at their goal locations.

    # Heuristic Initialization
    - Build a graph of locations based on the 'adjacent' predicates from static facts.
    - Extract the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the task definition.
    2. For the current state:
       - Identify the current location of each box that has a goal.
    3. Initialize the total heuristic value to 0.
    4. For each box that has a specified goal location:
       - Get the box's current location from the state.
       - If the box is not currently at its goal location:
         - Calculate the shortest path distance between the box's current location
           and its goal location using BFS on the pre-built location graph.
         - Add this distance to the total heuristic value. If no path exists,
           add a large penalty.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and storing box goals."""
        self.task = task # Store task for access to goals and static facts

        # Build the location graph from adjacent facts
        self.location_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            # Check for adjacent predicate and correct number of arguments (loc1, loc2, dir)
            if len(parts) == 4 and parts[0] == 'adjacent':
                loc1, loc2 = parts[1], parts[2] # Ignore direction for graph building
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = set()
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = set()
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Assuming adjacency is symmetric

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in task.goals:
            parts = get_parts(goal)
            # Goal facts can be complex, but for Sokoban, the relevant ones are (at box location)
            # Check for 'at' predicate and correct number of arguments (box, location)
            if len(parts) == 3 and parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 # Assume any object in an 'at' goal predicate is a box
                 self.box_goals[obj] = loc

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current locations of the boxes we care about (those with goals)
        current_box_locations = {}
        # We only need locations for boxes that have a goal
        boxes_to_track = set(self.box_goals.keys())

        for fact in state:
            parts = get_parts(fact)
            # Check for 'at' predicate and correct number of arguments (obj, location)
            # and if the object is one of the boxes we are tracking
            if len(parts) == 3 and parts[0] == 'at' and parts[1] in boxes_to_track:
                box, loc = parts[1], parts[2]
                current_box_locations[box] = loc

        total_heuristic = 0  # Initialize action cost counter.

        # Iterate through the boxes that have a goal
        for box, goal_location in self.box_goals.items():
            # Get the box's current location. If a box is not found in the state,
            # something is wrong, but we'll assume it's always present.
            current_location = current_box_locations.get(box)

            # If the box is not at its goal location
            if current_location != goal_location:
                # Calculate distance for this box
                # Ensure both current and goal locations are in the graph
                # (i.e., they are part of the movable area)
                if current_location in self.location_graph and goal_location in self.location_graph:
                     dist = bfs_distance(self.location_graph, current_location, goal_location)
                     # Add the distance. If no path exists (dist is 1000), add the penalty.
                     total_heuristic += dist
                else:
                     # If a location isn't in the graph, it's unreachable from the main area.
                     # This state is likely unsolvable or very bad. Assign a large penalty.
                     total_heuristic += 1000 # Large penalty

        # The heuristic is 0 if and only if all boxes are at their goal locations.
        # This is naturally handled by the loop: if all boxes are at their goals,
        # the 'if current_location != goal_location' condition is never met,
        # and total_heuristic remains 0.

        return total_heuristic
