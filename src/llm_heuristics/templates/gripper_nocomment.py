from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class GripperHeuristic(Heuristic):
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

    def __call__(self, node):
        state = node.state

        def match(fact, *args):
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        balls_in_room_a = sum(1 for fact in state if match(fact, "at", "*", "rooma"))

        balls_in_grippers = sum(1 for fact in state if match(fact, "carry", "*", "*"))

        robot_in_room_a = "(at-robby rooma)" in state

        move_cost = 1
        pick_cost = 1
        drop_cost = 1

        total_cost = 0

        if robot_in_room_a:
            if balls_in_grippers == 2:
                total_cost += move_cost + 2 * drop_cost
            elif balls_in_grippers == 1 and balls_in_room_a % 2 == 1:
                total_cost += pick_cost + move_cost + 2 * drop_cost
                balls_in_room_a -= 1
            elif balls_in_grippers == 1 and balls_in_room_a % 2 == 0:
                total_cost += move_cost + drop_cost
        else:
            total_cost += balls_in_grippers * drop_cost

        if balls_in_room_a > 0:
            total_cost += move_cost

            num_two_ball_trips = balls_in_room_a // 2

            total_cost += num_two_ball_trips * (2 * pick_cost + move_cost + 2 * drop_cost + move_cost) - 1

            if balls_in_room_a % 2 == 1:
                total_cost += move_cost + pick_cost + move_cost + drop_cost

        return total_cost
