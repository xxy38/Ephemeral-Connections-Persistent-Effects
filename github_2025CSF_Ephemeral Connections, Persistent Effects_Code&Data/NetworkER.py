import networkx as nx
import random
import math
from collections import deque

# ---------------------------
# Global Parameter Settings
# ---------------------------
# Network parameters
SIZE = 10000        # Number of nodes
R = 2               # R value, initial average degree of each node is 2*R
p = (2 * R) / (SIZE - 1)  # Corresponding probability in ER network to make average degree approximately 2R
# Game parameters
M = 10              # Memory length, i.e., records of the most recent M rounds
b = 1.2             # Payoff for a defector when playing against a cooperator (payoff is 1 when cooperating)
r = 0.5             # Cooperation ratio threshold: if the cooperation ratio in the past M rounds is below this value, a stranger connection is made
Ki = 3              # Number of strangers to randomly select and attempt to connect with in each round
alpha = 0.1         # Cost coefficient for stranger connections
steps_total = 10    # Total simulation rounds

# ---------------------------
# Define Agent class, each player records strategy, payoff, memory, and list of strangers
# ---------------------------
class Agent:
    def __init__(self):
        # Strategy: 0 for cooperator, 1 for defector; randomly assigned at initialization
        self.strategy = random.randrange(2)
        self.PreStrat = self.strategy    # Used to store the strategy of the previous round
        self.payoff = 0.0                # Current accumulated payoff
        self.memory = deque(maxlen=M)    # Deque to store decisions of the most recent M rounds (oldest record pops out automatically when full)
        self.mi = 0                      # Number of cooperation (strategy == 0) decisions recorded in memory
        self.strangers = []              # List of player IDs selected as stranger opponents in the current round

    def update_memory(self):
        """Adds the current round's decision to memory"""
        self.memory.append(self.strategy)

    def update_mi(self):
        """Updates the count of cooperation decisions in memory"""
        self.mi = sum(1 for decision in self.memory if decision == 0)

# Global player list, each network node corresponds to an Agent
Players = [Agent() for _ in range(SIZE)]

# Num is used to count global cooperators and defectors, Num[0] is cooperators, Num[1] is defectors
Num = [0, 0]

# ---------------------------
# Build an ER random network using NetworkX
# ---------------------------
def build_network(SIZE, R):
    # Edge probability p for ER network = average degree / (number of nodes - 1)
    avg_degree = 2 * R
    p = avg_degree / (SIZE - 1)
    G = nx.erdos_renyi_graph(SIZE, p)
    return G


G = build_network(SIZE, R)


# ---------------------------
# Initialize the state of all players
# ---------------------------
def initial():
    global Num, Players
    Num = [0, 0]
    for i in range(SIZE):
        Players[i].strategy = random.randrange(2)
        Players[i].PreStrat = Players[i].strategy
        Players[i].payoff = 0.0
        # Initialize memory and strangers (deque is already initialized as empty)
        Players[i].memory.clear()
        Players[i].strangers.clear()
        # Accumulate initial strategy distribution
        Num[Players[i].strategy] += 1

# ---------------------------
# Single player game function
# ---------------------------
def game(x):
    """
    Calculates player x's payoff in the current round:
    1. Game with fixed neighbors: iterate through all neighbors of x in the network.
    2. When memory is full (M rounds), if x's historical cooperation ratio is below threshold r,
       then randomly select Ki candidate stranger opponents (only check if their memory is full
       and cooperation ratio < r), and engage in a stranger game.
       Stranger game payoffs are also determined by strategy comparison, but incur a cost.
    """
    player = Players[x]
    payoff = 0.0
    strat = player.strategy

    # Game with fixed neighbors: iterate through all neighbors of x in the NetworkX graph
    for y in G.neighbors(x):
        if strat == 0 and Players[y].strategy == 0:
            payoff += 1
        elif strat == 1 and Players[y].strategy == 0:
            payoff += b

    # Stranger connections are only considered when memory is full (M rounds)
    if len(player.memory) == M:
        cooperation_ratio = player.mi / M
        ki = 0  # Records the number of stranger games played in the current round
        if cooperation_ratio < r:
            for _ in range(Ki):
                stranger = random.randrange(SIZE)
                # Only evaluate if the candidate's memory is full (M rounds)
                if len(Players[stranger].memory) == M:
                    stranger_ratio = Players[stranger].mi / M
                    if stranger_ratio < r:
                        if strat == 0 and Players[stranger].strategy == 0:
                            payoff += 1
                        elif strat == 1 and Players[stranger].strategy == 0:
                            payoff += b
                        ki += 1
                        player.strangers.append(stranger)
            # Stranger connections incur a cost: deduct b * alpha * (number of stranger participations)
            payoff -= b * alpha * ki
    return payoff

# ---------------------------
# Calculates the variation coefficient (CV) in the network
# ---------------------------
def calcCV():
    # First, count the number of "incoming links" from stranger connections for each node
    extra_in = [0] * SIZE
    for j in range(SIZE):
        for stranger in Players[j].strangers:
            extra_in[stranger] += 1
    degrees = [0.0] * SIZE
    total = 0.0
    for i in range(SIZE):
        # Effective degree of node i = number of fixed neighbors (G.degree(i))
        #                             + number of strangers actively selected by this node (len(Players[i].strangers))
        #                             + number of times selected by other nodes via stranger connections (extra_in[i])
        deg = G.degree(i) + len(Players[i].strangers) + extra_in[i]
        degrees[i] = deg
        total += deg
    mu_k = total / SIZE
    sq_diff_sum = sum((d - mu_k)**2 for d in degrees)
    sigma_k = math.sqrt(sq_diff_sum / SIZE)
    return sigma_k / mu_k

# ---------------------------
# Strategy update function
# ---------------------------
def change_strat(playerX, playerY):
    """
    When playerX and playerY compare their payoffs,
    if playerY's payoff is higher, playerX probabilistically changes
    its strategy to playerY's previous strategy.
    """
    global Num
    payoffX = Players[playerX].payoff
    payoffY = Players[playerY].payoff
    # Take the larger of the fixed neighbor counts of both as Kmax
    Kmax = max(G.degree(playerX), G.degree(playerY))
    if Kmax == 0:
        # Directly set probability to 0 or set Kmax to 1 to avoid division by zero
        probability = 0
    else:
        probability = (payoffY - payoffX) / (b * Kmax) if payoffY >= payoffX else 0
    if random.random() < probability:
        # Update global statistics: decrement count of original strategy and increment count of new strategy
        Num[Players[playerX].strategy] -= 1
        # Here playerX learns playerY's previous strategy PreStrat
        Players[playerX].strategy = Players[playerY].PreStrat
        Num[Players[playerX].strategy] += 1

# ---------------------------
# Memory update related
# ---------------------------
def update_memory(playerX):
    Players[playerX].update_memory()

def update_mi(playerX):
    Players[playerX].update_mi()

# ---------------------------
# Main program, executes the game simulation 
# ---------------------------
def main():
    random.seed()  # Initialize random seed based on current time
    initial()      # Initialize strategies and states of all players

    outfile = open("ER.txt", "w", encoding="utf8")

    for step in range(steps_total):
        # Synchronous update: reset payoff for each player, record previous strategy, and update memory and mi count
        for i in range(SIZE):
            Players[i].payoff = 0.0
            Players[i].PreStrat = Players[i].strategy
            update_memory(i)
            update_mi(i)

        # Calculate the game payoff for each player in the current round
        for i in range(SIZE):
            Players[i].payoff = game(i)

        # Strategy update phase
        for i in range(SIZE):
            playerX = i
            # Construct a list of potential players: first add fixed neighbors
            potentialPlayerY = list(G.neighbors(playerX))
            # If playerX's memory is full (M rounds) and its cooperation ratio is below r,
            # and its strangers list is not empty,
            # then also add the strangers with whom a game was played to the candidate list
            if len(Players[playerX].memory) == M and (Players[playerX].mi / M) < r and Players[playerX].strangers:
                potentialPlayerY.extend(Players[playerX].strangers)
            if potentialPlayerY:
                playerY = random.choice(potentialPlayerY)
                change_strat(playerX, playerY)

        # Calculate and output the Coefficient of Variation (CV) of degree and global cooperation ratio P_c for the current round
        CV = calcCV()
        P_c = Num[0] / SIZE
        line = f"{step}\tCV={CV:.4f}\tP_c={P_c:.4f}\n"
        print(line, end="")
        outfile.write(line)

        # Clear the stranger list recorded for each player at the end of each round
        for i in range(SIZE):
            Players[i].strangers.clear()

    outfile.close()

if __name__ == '__main__':
    main()


