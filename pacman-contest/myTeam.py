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
import random, time, util
from game import Directions
import game

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

    return

  def evaluate(self):
    return

  def getFeatures(self):
    return

  def getWeights(self):
    return

class getOffensiveActions(Actions):
  def __init__(self):
    return

  def getFeatures(self):
    #compute the distance to nearest boundary
    #compute score from successor state
    #compute the nearest food
    #compute the nearest capsule
    #compute the closet ghost
    return
  def getWeights(self):
    return

  def simulation(self):
    return

  def chooseAction(self):
    return

class getDefensiveActions(Actions):
  def __init__(self):
    return
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

class Attacker():

  def __init__(self):
    return

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
    self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

  def chooseAction(self, gameState):
    return

class Defender():
  def __init__(self, index):
    CaptureAgent.__init__(self, index)


  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
    self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

  def chooseAction(self, gameState):
    return