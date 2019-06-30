import run_simple
import run_rl
import sys
import matplotlib.pyplot as plot


iterations_rl  = run_rl.return_iterations()
plot.plot(iterations_rl)

iterations_simple  = run_simple.return_iterations()
plot.axhline(y=iterations_simple, color='r')

plot.title("Problem ID: " + str(sys.argv[1]) + "\nIterations per episode when goal is reached")
plot.xlabel("Episode")
plot.ylabel("Iterations")
plot.legend(["RL Agent","Simple Agent"],loc="upper right")
plot.savefig("Graph_"+str(sys.argv[1])+".png")
plot.show()
