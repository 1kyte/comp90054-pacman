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
import pacman


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
      if gameState != None:
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

  def startEpisode(self):
      """
        Called by environment when new episode is starting
      """
      self.lastState = None
      self.lastAction = None

class getOffensiveActions(Actions):
    def __init__(self, agent, index, gameState, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        self.agent = agent
        self.index = index
        self.weights = util.Counter()
        self.qValues = {}
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

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

    def doAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = state.getLegalActions(self.agent.index)
        randomAct = random.choice(legalActions)

        if len(legalActions) is 0:
            return None

        self.doAction(state, randomAct)
        return randomAct

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        actionReward = float('-inf')
        nextState = state.getLegalActions(self.index)
        for action in nextState:
            expectedQVal = self.getQValue(state, action)
            if actionReward < expectedQVal:
                actionReward = expectedQVal

        if actionReward == float('-inf'):
            return 0.0

        return actionReward

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

        features['carrying'] = successor.getAgentState(self.index).numCarrying

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
        return features

    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        numOfCarrying = successor.getAgentState(self.index).numCarrying
        opponents = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
        if len(visible) > 0:
            for agent in visible:
                if agent.scaredTimer > 0:
                    if agent.scaredTimer > 12:
                        return {'successorScore': 110, 'nearFood': -10, 'nearBoundary': 10-3*numOfCarrying, 'carrying': 350}

                    elif 6 < agent.scaredTimer < 12:
                        return {'successorScore': 110 + 5 * numOfCarrying, 'nearFood': -5, 'nearBoundary': -5-4*numOfCarrying, 'carrying': 100}

                # Visible and not scared
                else:
                    return {'successorScore': 110, 'nearFood': -10, 'nearBoundary': -15, 'carrying': 0}

        # Did not see anything
        self.counter += 1
        # print("Counter ",self.counter)
        return {'successorScore': 1000 + numOfCarrying * 3.5, 'nearFood': -7, 'nearBoundary': 5-numOfCarrying*3, 'carrying': 350}

    def simulation(self, depth, gameState, decay):
        new_state = gameState.deepCopy()
        if depth == 0:
            result_list = []
            action = random.choice(new_state.getLegalActions(self.index))
            next_state = new_state.generateSuccessor(self.index, action)
            result_list.append(self.evaluate(next_state, Directions.STOP))
            return max(result_list)

        # Get valid actions
        result_list = []
        actions = new_state.getLegalActions(self.index)

        # Randomly chooses a valid action
        for a in actions:
            # Compute new state and update depth
            next_state = new_state.generateSuccessor(self.index, a)
            result_list.append(
                self.evaluate(next_state, Directions.STOP) + decay * self.simulation(depth - 1, next_state, decay))
        return max(result_list)

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        finalValue = 0
        for key in self.getFeatures(state, action).keys():
            finalValue += self.weights[key] * self.getFeatures(state, action)[key]

        return finalValue

    def update(self, gameState, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        self.discount = 0.8
        self.alpha = 0.2
        self.reward = reward
        correction = (self.reward + (self.discount * self.getValue(nextState))) - self.getQValue(gameState, action)
        print correction
        for key in self.getFeatures(gameState, action).keys():
            self.weights[key] = self.weights[key] + self.alpha * correction * self.getFeatures(gameState, action)[key]


    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = 1
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def getLegalActions(self, state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)

    def actionFn(self, actionFn = None):

        if actionFn == None:
            actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn


    def observeTransition(self, state, action, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.update(state, action, nextState, deltaReward)

    def chooseAction(self, gameState):
        #start = time.time()
        # 3
        # Get valid actions. Randomly choose a valid one out of the best (if best is more than one)
        # All possible paths
        #actions = gameState.getLegalActions(self.agent.index)
        #actions.remove(Directions.STOP)

        #feasible = []
        #for a in actions:
            #value = 0
            # for i in range(0, 10):
            #     value += self.randomSimulation1(2, new_state, 0.8) / 10
            # fvalues.append(value)

            #self.simulation(2, gameState.generateSuccessor(self.agent.index, a), 0.7)
        self.startEpisode()
        a = self.getAction(gameState)
        self.observationFunction(gameState)
        self.final(gameState)
        # self.getAction(gameState)
        # self.observationFunction(gameState)
        # self.final(gameState)
            #feasible.append(value)
        return a
        #bestAction = max(feasible)
        #possibleChoice = filter(lambda x: x[0] == bestAction, zip(feasible, actions))
        # print 'eval time for offensive agent %d: %.4f' % (self.agent.index, time.time() - start)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        deltaReward = 1
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

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
    # for a in actions:
    #     self.evaluate(gameState, a)
    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

    def getWeights(self, gameState, action):
        """
                The weights and features contain:
                    1. the distance between ghost and boundary;
                    2. the distance between the ghost and the pacdot which is the clasest point with boundary;
                    3. the distance between the ghost and enemy.
                    4. the distance between the ghost and the protected Capsule
            """

        s = self.getSuccessor(gameState, action)

        # return

    def getFeatures(self, gameState, action):
        features = util.Counter()

        # get the distance to boundary

        # get the distance to the food
        protectedFood = self.agent.getFoodYouAreDefending(gameState)

        # get the distance to enemy

        # get the distance to protected Capsule
        capsulesList = self.agent.getCapsulesYouAreDefending(gameState)

class Attacker(CaptureAgent):

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    #self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
    self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

  def chooseAction(self, gameState):
      #2
      # self.enemies = self.getOpponents(gameState)
      #invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]

      #if self.getScore(gameState) >= 13:
      #    return self.DefenceStatus.chooseAction(gameState)
      #else:
      # self.OffenceStatus.getAction(gameState)
      return self.OffenceStatus.chooseAction(gameState)
      # self.OffenceStatus.final(gameState)
      #random.choice(gameState.deepCopy().getLegalActions(self.index))
      #self.enemies = self.getOpponents(gameState)
      #invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]


class Defender(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)


  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
    #self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

  def chooseAction(self, gameState):
    return self.DefenceStatus.chooseAction(gameState)


