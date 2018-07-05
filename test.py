from puzzle import *

difficulty = 18

goal = parse_state(goal1)
algos = [astar_pattern,astar]

if __name__ == "__main__":
    train(goal)

    print "Training finishes."

    game = generate(goal, difficulty)

    print game
    solutions = [solve(game, goal, algo) for algo in algos]

    for solution in solutions:
        print solution

    print "----------------"
