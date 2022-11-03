from random import randint, shuffle

def land_rocket(plan): # input argument is vector of '0's or '1's defining if rocket engine is working or not in given step
  """A method that simulates rocket landing and returns gain of mission"""
  
  # constants
  LANDING_PRIZE = 2000
  CRASH_PENALTY = -1000
  LANDING_HEIGHT = 2
  LANDING_VELOCITY = 2
  CRASH_HEIGHT = 0
  ACC_FACTOR = 45
  G = -0.009
  
  # initial parameters of mission
  h = 200; v = 0; m = 200; a = 0
  
  # initial gain of mission
  gain = 0
  
  # incrementing initial rocket mass for every step which requires working engine
  for unit in plan:
    m += unit
    gain -= unit
  
  for i in range(len(plan)):
    if h < LANDING_HEIGHT and abs(v) < LANDING_VELOCITY:
      gain += LANDING_PRIZE
      return gain
    elif h < CRASH_HEIGHT:
      gain += CRASH_PENALTY
      return gain
    else:
      if plan[i]:
        m -= 1
        a = ACC_FACTOR/m
      v += (a + G)
      h -= v
  
  return gain

def generate_population(chrom_size, pop_size):
  """A method that generates initial population of given size for the problem"""
  
  population = []
  
  for _ in range(pop_size):
    zeros_count = randint(0, chrom_size)
    ones_count = chrom_size - zeros_count
    chrom = [0]*zeros_count + [1]*ones_count
    shuffle(chrom)
    population.append(chrom)
  
  return population
  
  