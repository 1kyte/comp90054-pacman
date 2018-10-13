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
import random, util, time, math
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
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
          # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
    else:
        return successor

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self):
      """
      Normally, weights do not depend on the gamestate.  They can be either
      a counter or a dictionary.
      """
      return {'nearFood': -10, 'GhostDistance': 0, 'distanceToEnemiesPacMan': 0, 'successorScore':100}
      #return {'nearFood': -7, 'successorScore':100}

  def getValues(self, gameState, action):
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


  def observeTransition(self, state, action, nextState):
      """
          Called by environment to inform agent that a transition has
          been observed. This will result in a call to self.update
          on the same arguments

          NOTE: Do *not* override or call this function
      """
      self.update(state, action, nextState)

  def observationFunction(self, state):
      """
          This is where we ended up after our last action.
          The simulation should somehow ensure this is called
      """
      if not self.lastState is None:
          # reward = 1
          self.observeTransition(self.lastState, self.lastAction, state)
      return state

  def getDirectedAction(self,gameState):

      isWall = True
      changed = False
      target = None
      # selfPos = gameState.getAgentState(self.index).getPosition()
      foodCount = self.agent.getFood(gameState).asList()
      x,y = gameState.getAgentState(self.index).getPosition()
      selfx, selfy = self.initPos
      if not self.t == None:
        tx ,ty= self.t
      if x == selfx and y == selfy:
          changed = True

      if self.isAttacker:
          if self.count.has_key(self.t):
              if self.count[self.t] > 4 and abs(tx-x)<2:
                  target = foodCount[0]
                  self.t = target
                  self.count[self.t]=0

      while isWall:

          if changed:
              target = random.choice(self.boundary)
              self.t = target

          else:
              target = self.t

          if self.isAttacker:
              if self.count.has_key(target):
                  self.count[target] += 1
              else:
                  self.count[target] = 1
          isWall = gameState.data.layout.isWall(self.t)

      legalActionList = gameState.getLegalActions(self.agent.index)

      legalActionList.remove(Directions.STOP)

      mindistance = 9999999
      act = None
      for a in legalActionList:
          successor = self.getSuccessor(gameState,a)
          successPos = successor.getAgentState(self.index).getPosition()
          dis = self.agent.getMazeDistance(successPos,target)
          if dis<mindistance:
              mindistance = dis
              act = a
      return act

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

# ==============================================================================================================
# Offence Action
#
# ==============================================================================================================

class getOffensiveActions(Actions):
    def __init__(self, agent, index, gameState, actionFn = None, epsilon=0.5, alpha=0.5, gamma=1):
        self.agent = agent
        self.index = index
        self.weights = util.Counter()
        self.qValues = {}
        self.reward = 1
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.initPos = gameState.getAgentState(self.index).getPosition()
        self.isAttacker = True
        self.t = None
        self.count = {}
        self.weights = self.getWeights()
        self.startEpisode()

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

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        if len(state.getLegalActions(self.index)) == 0:
            return None

        maxActionValue = float('-inf')
        maxAction = None

        maxLegalActions = state.getLegalActions(self.agent.index)
        maxLegalActions.remove(Directions.STOP)
        maxReversed = Directions.REVERSE[state.getAgentState(self.agent.index).configuration.direction]
        if not self.lastAction == None:
            if maxReversed in maxLegalActions and len(maxLegalActions) > 1:
                maxLegalActions.remove(maxReversed)
        for action in maxLegalActions:
            val = self.getQValue(state, action)
            if maxActionValue == val:
                randPolicy = random.choice([maxAction, action])
                if randPolicy == action:
                    maxAction = action
                    maxActionValue = val
            elif maxActionValue < val:
                maxActionValue = val
                maxAction = action
        return maxAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

        """
        # Pick Action
        legalActions = state.getLegalActions(self.agent.index)
        legalActions.remove(Directions.STOP)
        reversed = Directions.REVERSE[state.getAgentState(self.agent.index).configuration.direction]
        if not self.lastAction == None:
            if reversed in legalActions and len(legalActions)>1:
                legalActions.remove(reversed)

        isHeads = util.flipCoin(self.epsilon)
        randomAct = random.choice(legalActions)

        if len(legalActions) is 0:
            return None

        if isHeads:
            self.doAction(state, randomAct)
            return randomAct
        else:
            policyAct = self.getPolicy(state)
            if policyAct is None:
                policyAct = randomAct
                policyAct.remove(Directions.STOP)
            self.doAction(state, policyAct)
            return policyAct

    def doAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

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
            #print 'expectedQVal', expectedQVal
            if actionReward < expectedQVal:
                actionReward = expectedQVal

        if actionReward == float('-inf'):
            return 0.0

        #print 'actionReward', actionReward
        return actionReward

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # get current position of the agent
        currentPosition = successor.getAgentState(self.index).getPosition()

        # compute the distance to nearest boundary
        # currentDistance = self.agent.getMazeDistance(currentPosition, self.boundary[0])
        #
        # for pos in range(len(self.boundary)):
        #     distance = self.agent.getMazeDistance(currentPosition, self.boundary[pos])
        #     if (currentDistance > distance):
        #         currentDistance = distance
        # features['nearBoundary'] = float(currentDistance) / (gameState.getWalls().width * gameState.getWalls().height)

        # features['carrying'] = successor.getAgentState(self.index).numCarrying

        #compute the nearest food
        foodCount = self.agent.getFood(successor).asList()
        # if len(foodCount) > 0:
        #     currentFoodDis = 999999
        #     for food in foodCount:
        #         disFood = self.agent.getMazeDistance(currentPosition, food)
        #         if (disFood < currentFoodDis):
        #             currentFoodDis = disFood
        #     features['nearFood'] = currentFoodDis
        #
        #compute the nearest capsule
        # capsuleList = self.agent.getCapsules(successor)
        # if len(capsuleList) > 0:
        #     minCapsuleDistance = 99999
        #     for c in capsuleList:
        #         distance = self.agent.getMazeDistance(currentPosition, c)
        #         if distance < minCapsuleDistance:
        #             minCapsuleDistance = distance
        #     features['distanceToCapsule'] = minCapsuleDistance
        # else:
        #     features['distanceToCapsule'] = 0

        #compute the closet ghost
        opponentsState = []
        for i in self.agent.getOpponents(successor):
            opponentsState.append(successor.getAgentState(i))
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponentsState)
        if len(visible) > 0:
            positions = [agent.getPosition() for agent in visible]
            closest = min(positions, key=lambda x: self.agent.getMazeDistance(currentPosition, x))
            closestDist = self.agent.getMazeDistance(currentPosition, closest)
            if closestDist <= 5:
                # print(CurrentPosition,closest,closestDist)
                features['GhostDistance'] = closestDist

        else:
            probDist = []
            for i in self.agent.getOpponents(successor):
                probDist.append(successor.getAgentDistances()[i])
                features['GhostDistance'] = min(probDist)

        features['successorScore'] = -len(foodCount)

        enemiesPacMan = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        Range = filter(lambda x: x.isPacman and x.getPosition() != None, enemiesPacMan)
        if len(Range) > 0:
            positions = [agent.getPosition() for agent in Range]
            closest = min(positions, key=lambda x: self.agent.getMazeDistance(currentPosition, x))
            closestDist = self.agent.getMazeDistance(currentPosition, closest)
            if closestDist < 4:
                 features['distanceToEnemiesPacMan'] = closestDist
        else:
            features['distanceToEnemiesPacMan'] = 0

        features.divideAll(10.0)
        return features

    def getReward(self, gameState, action):
        """
          Return the reward of action
          eat food = 1 point
          carry food home = 100 * n point, n is number of carrying food
          be caught = -10 * n point, n is number of carrying food
        """


        # successor = gameState.generateSuccessor(gameState, action)
        successor = self.getSuccessor(gameState, action)
        # currentCarrying = gameState.numCarring
        currentCarrying = gameState.getAgentState(self.index).numCarrying
        nextCarrying = successor.getAgentState(self.index).numCarrying
        state = successor.getAgentState(self.index).isPacman
        reward = self.reward
        currentPos = gameState.getAgentState(self.index).getPosition()
        nextPos = successor.getAgentState(self.index).getPosition()
        feature = self.getFeatures(gameState, action)

        nextDistance = 99999
        for pos in range(len(self.boundary)):
            distance = self.agent.getMazeDistance(nextPos, self.boundary[pos])
            if (nextDistance > distance):
                nextDistance = distance
        drawback = nextDistance

        currentDistance = 99999
        for pos in range(len(self.boundary)):
            distance = self.agent.getMazeDistance(currentPos, self.boundary[pos])
            if (currentDistance > distance):
                currentDistance = distance

        capsuleList = self.agent.getCapsules(successor)
        if len(capsuleList) > 0:
            minCapsuleDistance = 99999
            for c in capsuleList:
                distance = self.agent.getMazeDistance(currentPos, c)
                if distance < minCapsuleDistance:
                    minCapsuleDistance = distance
        else:
            minCapsuleDistance = 0

        #next state is pacman
        if state:
            if not gameState.getAgentState(self.index).isPacman:
                reward = 1
            else:
            # reward = 10
            #reward = reward + nextCarrying - currentCarrying
                if currentCarrying > 5:
                    reward = reward - drawback
                enemies = [gameState.getAgentState(i) for i in self.agent.getOpponents(gameState)]
                inRange = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
                opponents = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
                visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
                # print action
                if len(visible) > 0:
                    for agent in visible:
                        if agent.scaredTimer < 5 and agent.scaredTimer >= 0:
                            if agent.getPosition() == nextPos:
                                reward = -300
                            else:
                                nextToEmenies = self.agent.getMazeDistance(nextPos, agent.getPosition())
                                currentToEmenies = self.agent.getMazeDistance(currentPos, agent.getPosition())
                                if nextToEmenies < currentToEmenies:
                                    reward = -300
                                else:
                                    reward = 50
                        else:
                            if minCapsuleDistance < feature['GhostDistance']:
                                if agent.scaredTimer >= 0:
                                    reward = 20
                            if len(inRange) > 0:
                                currentInvaderDis = self.agent.getMazeDistance(currentPos, agent.getPosition())
                                nextInvaderDis, nextInvaderPac = min(
                                    [(self.agent.getMazeDistance(nextPos, x.getPosition()), x) for x in inRange])
                                if currentInvaderDis <= nextInvaderDis:
                                    reward = 50
                                else:
                                    reward = -50
                            if agent.getPosition() == nextPos:
                                reward = -300
                x,y = nextPos
                if self.agent.red:
                    if x == 1:
                        reward = -500
                    else:
                        reward = 5
                else:
                    limit = self.agent.getFood(gameState).width-2
                    if x == limit:
                        reward = -500
                    else:
                        reward = 5



        #nextstate is not pacman
        else:
            # currentPos = gameState.getPosition()
            if nextCarrying > currentCarrying:
                reward = 10
            if not gameState.getAgentState(self.index).isPacman:
                if nextDistance<drawback:
                    reward = 11
                else:
                    reward = -1
            else:
                reward = 200
            move = self.agent.getMazeDistance(currentPos, nextPos)
            #eaten
            if move > 1:
                reward = -200 * currentCarrying
            else:
                if currentDistance >  nextDistance:
                    reward = 5 * currentCarrying
                if gameState.getAgentState(self.index).isPacman and currentCarrying >0:
                    reward = 100 * currentCarrying
            #else:
        #print 'reward', reward
        return reward

    def getQValue(self, gameState, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        #self.weights = self.getWeights(state, action)
        finalValue = 0
        for key in self.getFeatures(gameState, action).keys():
            finalValue += self.weights[key] * self.getFeatures(gameState, action)[key]
        return finalValue

    def update(self, gameState, action, nextState):
        """
        Should update your weights based on transition
        """
        self.discount = 0.8
        self.alpha = 0.2
        self.reward = self.getReward(gameState,action)
        a = self.getValue(nextState)
        b = self.getQValue(gameState, action)
        #successor = self.getSuccessor(gameState, action)
        #print 'successor score', self.agent.getScore(successor)

        correction = (self.reward + (self.discount * self.getValue(nextState))) - self.getQValue(gameState, action)
        # print 'reward', self.reward
        # print 'random', self.discount * self.getValue(nextState)
        # print correction
        for key in self.getFeatures(gameState, action).keys():
            self.weights[key] = self.weights[key] + self.alpha * correction * self.getFeatures(gameState, action)[key]

        # print "offence weights" , self.weights
        # print "offence featrures", self.getFeatures(gameState, action)
        #print "correction %.2f" % correction
        #if math.isnan(self.weights['nearFood']):
        #    print "is nan"
        # if self.weights['nearBoundary']>50:
        #     print "too large"

    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            # reward = 1
            self.observeTransition(self.lastState, self.lastAction, state)
        return state


    def chooseAction(self, gameState):
        #start = time.time()
        # 3
        # Get valid actions. Randomly choose a valid one out of the best (if best is more than one)
        # All possible paths
        #actions = gameState.getLegalActions(self.agent.index)
        #actions.remove(Directions.STOP)

        x, y = gameState.getAgentState(self.index).getPosition()
        bx,by = self.boundary[0]
        if not self.agent.red:
            if x < bx:
                a = self.getAction(gameState)
                self.observationFunction(gameState)
                self.final(gameState)
                # return (a, self.weights)
                return a
            else:
                # return (self.getDirectedAction(gameState),self.weights)
                return self.getDirectedAction(gameState)
        else:
            if x > bx:
                a = self.getAction(gameState)
                self.observationFunction(gameState)
                self.final(gameState)
                # return (a, self.weights)
                return a
            else:
                # return (self.getDirectedAction(gameState), self.weights)
                return self.getDirectedAction(gameState)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        # deltaReward = 1
        self.observeTransition(self.lastState, self.lastAction, state)
        #print "weights", self.weights

# ==============================================================================================================
# Defencd Action
#
# ==============================================================================================================

class getDefensiveActions(Actions):
  def __init__(self, agent, index, gameState,actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
    self.index = index
    self.agent = agent
    self.DenfendList = {}
    self.preTarget = None
    self.epsilon = float(epsilon)
    if actionFn == None:
        actionFn = lambda state: state.getLegalActions()
    self.actionFn = actionFn
    self.numTraining = numTraining
    self.alpha = float(alpha)
    self.discount = float(gamma)
    # self.weights = util.Counter()
    self.reward = 5.0
    self.protectedFoods = self.getFoodPosition(gameState)
    self.isAttacker = False
    self.t = None
    self.target = None
    self.initPos = gameState.getAgentState(self.index).getPosition()
    self.weights = {'distToBound': 5, 'distToEnemy': 5, 'distToCapsule': 2}  # 'distToTarget': 5,
    self.chase=False
    self.startEpisode()

    if self.agent.red:
      middle = (gameState.data.layout.width - 2) / 2
    else:
      middle = ((gameState.data.layout.width - 2) / 2) + 1
    self.boundary = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(middle, i):
        self.boundary.append((middle, i))

    self.target = None
    self.lastObservedFood = None


  def doAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

  def chooseAction(self, gameState):
      x,y = gameState.getAgentState(self.index).getPosition()
      if not self.agent.red:
        if x<=30:
            action = self.getAction(gameState)
            self.observationFunction(gameState)
            self.final(gameState)
            # return (action, self.weights)
            return action
        else:
            # return (self.getDirectedAction(gameState),self.weights)
            return self.getDirectedAction(gameState)
      else:
         if x >= 2:
             action = self.getAction(gameState)
             self.observationFunction(gameState)
             self.final(gameState)
             # return (action, self.weights)
             return action
         else:
             # return (self.getDirectedAction(gameState),self.weights)
             return self.getDirectedAction(gameState)

  def getAction(self,gameState):
      legalActionList = gameState.getLegalActions(self.agent.index)
      legalActionList.remove(Directions.STOP)

      reversed = Directions.REVERSE[gameState.getAgentState(self.agent.index).configuration.direction]
      if not self.lastAction == None:
          if reversed in legalActionList and len(legalActionList) > 1:
              legalActionList.remove(reversed)

      randomAct = random.choice(legalActionList)
      if len(legalActionList) is 0:
          return None

      policyAct = self.getPolicy(gameState)
      if policyAct is None:
          policyAct = randomAct
          # if the ghost become pacman in the next step this action should be removed
      self.doAction(gameState, policyAct)
      return policyAct

  def getPolicy(self, state):
      """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
      """
      if len(state.getLegalActions(self.index)) == 0:
          return None

      maxActionValue = float('-inf')
      maxAction = None

      currentState = self.agent.getCurrentObservation()

      maxLegalActions = state.getLegalActions(self.agent.index)
      maxLegalActions.remove(Directions.STOP)

      d = 9999999
      for action in maxLegalActions:
          successor = self.getSuccessor(state, action)
          successorPos = successor.getAgentState(self.index).getPosition()
          sx,sy = successorPos
          bx,by = self.boundary[0]
          if self.agent.red:
            if bx<sx:
                inBoundary = False
            else:
                inBoundary = True
          else:
              if sx <bx:
                  inBoundary = False
              else:
                  inBoundary = True

          val = self.getQValue(state, action)

          if inBoundary:
              enemies = [currentState.getAgentState(i) for i in self.agent.getOpponents(state)]
              visible = filter(lambda enemy: enemy.isPacman and enemy.getPosition() != None, enemies)
              farEnemies = [currentState.getAgentState(i) for i in self.agent.getOpponents(successor)]
              farVisible = filter(lambda enemy: enemy.isPacman and enemy.getPosition() != None, farEnemies)
              if len(visible)>0:
                self.chase = False
                self.target = None
                if state.getAgentState(self.index).scaredTimer == 0:

                    for agent in visible:
                        dis = self.agent.getMazeDistance(successorPos,agent.getPosition())
                        if successorPos == agent.getPosition():
                            self.chase = False
                            self.target = None
                            return action
                        elif dis<d:
                            # self.chase = True
                            self.chase = False
                            self.target = None
                            d = dis
                            maxAction = action
                elif len(farVisible)>0:
                    for agent in farVisible:
                        dis = self.agent.getMazeDistance(successorPos,agent.getPosition())
                        if dis<d:
                            self.chase = True
                            d = dis
                            maxAction = action
                else:
                    maxAction, maxActionValue = self.usualPolicy(state,maxActionValue,action,val,maxAction,successor)

              else:
                  maxAction, maxActionValue = self.usualPolicy(state,maxActionValue, action, val, maxAction,successor)

          else:
              legalActions = state.getLegalActions(self.agent.index)
              legalActions.remove(Directions.STOP)
              legalActions.remove(action)
              maxAction = random.choice(legalActions)
      return maxAction

  def getFoodPosition(self,state):
      foodList = self.agent.getFoodYouAreDefending(state)
      foods = []
      for x in range(foodList.width):
          for y in range(foodList.height):
              if foodList[x][y]:
                foods.append((x,y))
      # print foods
      return foods

  def getTempTarget(self,state):
      foodList = self.agent.getFoodYouAreDefending(state)
      # self.protectedFoods = self.getFoodPosition(state)
      # foods = []
      # for x in range(foodList.width):
      #     for y in range(foodList.height):
      #         if foodList[x][y]:
      #             foods.append((x, y))
      selfPos = state.getAgentState(self.index).getPosition()
      thePos = None
      for food in self.protectedFoods:
          x,y = food
          if not foodList[x][y]:
              self.target = (x,y)
              self.chase = True
              self.protectedFoods = self.getFoodPosition(state)
              return self.target
          else:
              if selfPos == self.target:
                  self.chase = False

              if self.chase:
                  if not self.target==None:
                      return self.target
                  else:
                      thePos = self.boundary[len(self.boundary) / 2]
              else:
                  thePos = self.boundary[len(self.boundary)/2]

              if selfPos == self.boundary[len(self.boundary)/2]:
                  self.protectedFoods = self.getFoodPosition(state)

      return thePos



  def usualPolicy(self,state,maxActionValue,action,val,maxAction,successor):

      tempTarget = self.getTempTarget(state)
      successorPos = successor.getAgentState(self.index).getPosition()
      succToTarDistance = self.agent.getMazeDistance(tempTarget,successorPos)
      rate = 1
      if not succToTarDistance==0:
        rate = 1/float(succToTarDistance)
      val = rate*val
      if maxActionValue == val:
          randPolicy = random.choice([maxAction, action])
          if randPolicy == action:
              maxAction = action
              maxActionValue = val
      if maxActionValue < val:
          maxActionValue = val
          maxAction = action
      return (maxAction,maxActionValue)


  def getFeatures(self, gameState, action):
        features = util.Counter()
        myPos = gameState.getAgentState(self.index).getPosition()

        # get the distance to boundary
        minDisToB = 10000
        for position in self.boundary:
            distance = self.agent.getMazeDistance(myPos, position)
            if distance < minDisToB:
                minDisToB = distance
        features['distToBound'] = float(minDisToB) / (gameState.getWalls().width * gameState.getWalls().height)

        # # get the distance to the food
        # defendTarget = self.getDenfendTarget(gameState)
        # distToTarget = self.agent.getMazeDistance(myPos, defendTarget)
        # features['distToTarget'] = distToTarget


        # get the distance to enemy
        enemies = [gameState.getAgentState(i) for i in self.agent.getOpponents(gameState)]
        visible = filter(lambda enemy: enemy.isPacman and enemy.getPosition() != None, enemies)
        if len(visible) > 0:
            enemyDistance,state = min([(self.agent.getMazeDistance(myPos, enemy.getPosition()), enemy) for enemy in visible])
        else:
            enemyDistance = 1
        if enemyDistance==0:
            features['distToEnemy'] = 0
        else:
            features['distToEnemy'] = 1/float(enemyDistance)

        # get the distance to protected Capsule
        capsulesList = self.agent.getCapsulesYouAreDefending(gameState)
        if len(capsulesList) >0:
            minDisToC = 10000
            for capsule in capsulesList:
                distance = self.agent.getMazeDistance(myPos, capsule)
                if distance < minDisToC:
                    minDisToC = distance
            if minDisToC ==0:
                features['distToCapsule'] = 0
            else:
                features['distToCapsule'] = 1/float(minDisToC)
        else:
            features['distToCapsule'] = 0

        features.divideAll(10.0)
        return features

  def getDenfendTarget(self, gameState):
      if self.preTarget == None:
          defFoodList = self.agent.getFoodYouAreDefending(gameState).asList()
          nearestFood = None
          minDistance = 10000
          for food in defFoodList:
              minDisToB = 10000
              for position in self.boundary:
                  distance = self.agent.getMazeDistance(food, position)
                  if distance < minDisToB:
                      minDisToB = distance
              if minDisToB < minDistance:
                  nearestFood = food
          return nearestFood
      else:
          currentDefendingFood = self.agent.getFoodYouAreDefending(gameState).asList
          if len(self.preDefendingFood) == len(currentDefendingFood):
              return self.preTarget
          else:
              foodEatenList = list(set(self.preDefendingFood).difference(set(currentDefendingFood)))
              foodEaten = random.choice(foodEatenList)
              nextEatenDistance = 10000
              nextEaten = None
              for food in currentDefendingFood:
                  distance = self.agent.getMazeDistance(food, foodEaten)
                  if distance < nextEatenDistance:
                      nextEatenDistance = distance
                      nextEaten = food
              return nextEaten

  def final(self,gamestate):
      self.observeTransition(self.lastState, self.lastAction, gamestate)
      # self.stopEpisode()
      return

  def getValues(self, gameState):
      actionReward = float('-inf')
      nextState = gameState.getLegalActions(self.index)
      for action in nextState:
          expectedQVal = self.getQValue(gameState, action)
          # print 'expectedQVal', expectedQVal
          if actionReward < expectedQVal:
              actionReward = expectedQVal

      if actionReward == float('-inf'):
          return 0.0

      return actionReward

  def getQValue(self,gameState, action):
      # self.weights = self.getWeights(state, action)
      finalValue = 0
      for key in self.getFeatures(gameState, action).keys():
          finalValue += self.weights[key] * float(self.getFeatures(gameState, action)[key])

      return finalValue

  def update(self,state, action, nextState):
      reward = self.getRewards(state,action)
      self.reward = reward
      feature = self.getFeatures(state,action)

      self.discount = 0.8
      self.alpha = 0.2

      correction = (reward + (self.discount * self.getValues(nextState))) - self.getQValue(state, action)

      # print correction
      for key in feature.keys():
          self.weights[key] = self.weights[key] + self.alpha * correction * feature[key]

      # print 'reward',reward
      # print 'feature',feature
      # print 'weight',self.weights

  def getRewards(self, gameState, action):
      # reward = 5.0
      successor = self.getSuccessor(gameState, action)
      currentPos = gameState.getAgentState(self.index).getPosition()
      nextPos = successor.getAgentState(self.index).getPosition()
      feature = self.getFeatures(gameState, action)
      state = successor.getAgentState(self.index).isPacman
      currentState = gameState.getAgentState(self.index).isPacman

      if currentState:
          reward = -500.0
      else:
          nextDistance = self.agent.getMazeDistance(nextPos, self.boundary[0])
          for pos in range(len(self.boundary)):
              distance = self.agent.getMazeDistance(nextPos, self.boundary[pos])
              if (nextDistance > distance):
                  nextDistance = distance
          currentDistance = feature['distToBound']
          if nextDistance<currentDistance:
              reward = 5.0
          else:
              reward = self.reward - currentDistance

      enemies = [gameState.getAgentState(i) for i in self.agent.getOpponents(gameState)]
      inRange = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
      opponents = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
      visible = filter(lambda x: x.isPacman and x.getPosition() != None, opponents)
      if len(visible) > 0:
          for agent in visible:
              nextToEnemies = self.agent.getMazeDistance(nextPos,agent.getPosition())
              currentToEnemies = self.agent.getMazeDistance(currentPos,agent.getPosition())
              if gameState.getAgentState(self.index).scaredTimer > 3:
                  if nextToEnemies<currentToEnemies:
                      reward = -5
                  else:
                      reward = 5
              else:
                  if nextToEnemies<currentToEnemies:
                      reward = 100
                  else:
                      reward = -50


      return reward


# ==============================================================================================================
# Attacker Agent
# ==============================================================================================================

class Attacker(CaptureAgent):

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    #self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
    self.OffenceStatus = getOffensiveActions(self, self.index, gameState)
    # if self.weights == {}:
    #     self.weights = self.OffenceStatus.getWeights()
    # if not self.weights == {}:
    #     self.OffenceStatus.weights = self.weights

  def chooseAction(self, gameState):
      #2
      # a, self.weights = self.OffenceStatus.chooseAction(gameState)
      a = self.OffenceStatus.chooseAction(gameState)
      return a
# =======================================================
# Defender Agent
#
# =======================================================

class Defender(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
    #self.OffenceStatus = getOffensiveActions(self, self.index, gameState)
    # if self.weights == {}:
    # self.weights = {'distToBound': 5,  'distToEnemy': 5, 'distToCapsule': 2}#'distToTarget': 5,
    # if not self.weights == {}:
    #     self.DefenceStatus.weights = self.weights

  def chooseAction(self, gameState):
    # return Directions.STOP
    a = self.DefenceStatus.chooseAction(gameState)
    return a