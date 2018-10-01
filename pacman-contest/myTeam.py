# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, util, time
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Attacker', second = 'Defender'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class Actions():

  def getSuccessor(self, gameState, action):
      successor = gameState.generateSuccessor(self.index, action)
      pos = successor.getAgentState(self.index).getPosition()
      if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
      else:
          return successor

  def evaluate(self, gameState, action):
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class getOffensiveActions(Actions):
    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index
        #self.agent.distancer.getMazeDistances()
        self.counter = 0

        if self.agent.red:
            boundary = (gameState.data.layout.width - 2) / 2

        if not self.agent.red:
            boundary = ((gameState.data.layout.width - 2) / 2) + 1

        self.boundary = []

        for middlePoint in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary, middlePoint):
                self.boundary.append((boundary, middlePoint))

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Compute score from successor state
        features['successorScore'] = self.agent.getScore(successor)

        # get current position of the agent
        currentPosition = successor.getAgentState(self.index).getPosition()

        # compute the distance to nearest boundary
        currentDistance = self.agent.getMazeDistance(currentPosition, self.boundary[0])

        for pos in range(len(self.boundary)):
            distance = self.agent.getMazeDistance(currentPosition, self.boundary[pos])
            if (currentDistance > distance):
                currentDistance = distance
        features['nearBoundary'] = currentDistance

        #compute the nearest food
        foodCount = self.agent.getFood(successor).asList()
        if len(foodCount) > 0:
            currentFoodDis = self.agent.getMazeDistance(currentPosition, foodCount[0])
            for food in foodCount:
                disFood = self.agent.getMazeDistance(currentPosition, food)
                if (disFood < currentFoodDis):
                    currentFoodDis = disFood
            features['nearFood'] = currentFoodDis

    #compute the nearest capsule
    #compute the closet ghost
        return
    def getWeights(self):
        return

    def simulation(self):
        return

    def chooseAction(self, gameState):
        start = time.time()
        actions = gameState.getLegalActions(self.agent.index)
        return random.choice(actions)

class getDefensiveActions(Actions):
  def __init__(self, agent, index, gameState):
    self.index = index
    self.agent = agent
    self.DenfendList = {}

    if self.agent.red:
      middle = (gameState.data.layout.width - 2) / 2
    else:
      middle = ((gameState.data.layout.width - 2) / 2) + 1
    self.boundary = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(middle, i):
        self.boundary.append((middle, i))

  def defenceProbability(self):
    return
  def selectPatrolTarget(self):
    return
  def chooseAction(self, gameState):
    """
        Picks among actions randomly.
        """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

class Attacker(CaptureAgent):

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    #self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
    self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

  def chooseAction(self, gameState):
      random.choice(gameState.deepCopy().getLegalActions(self.index))
      #self.enemies = self.getOpponents(gameState)
      #invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]


class Defender(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)


  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    #self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
    #self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

  def chooseAction(self, gameState):
    return