#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import epydemic as ep
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.special import zeta
from mpmath import polylog as Li   # use standard name
import mpmath


################################
################################ SIR
################################

class MonitoredSIR(ep.SIR):
    INTERVAL = 'interval'
    PROGRESS = 'progress'

    def setUp(self, params):
        '''Schedule the monitoring event.

        :param params: the simulation parameters'''
        super(MonitoredSIR, self).setUp(params)

        # add a monitoring event to fill-in the evolution of the process
        self._series = []
        self.postRepeatingEvent(0, params[self.INTERVAL], None, self.monitor)

    def monitor(self, t, e):
        '''Record the sizes of each compartment.

        :param t: the simulation time
        :param e: the element (ignored)'''
        s = dict()
        for k in [ep.SIR.SUSCEPTIBLE, ep.SIR.INFECTED, ep.SIR.REMOVED]:
            s[k] = len(self.compartment(k))
        self._series.append((t, s))

    def results(self):
        '''Add the time series to the experiment's results.

        :returns: a results dict including the monitored time series'''
        rc = super(MonitoredSIR, self).results()

        rc[self.PROGRESS] = self._series
        return rc


################################
################################ Network
################################

def generateFrom(N, p, maxdeg = 1000):
    '''Generate a random graph with degree distribution described
    by a model function.

    :param N: number of numbers to generate
    :param p: model function
    :param maxdeg: maximum node degree we'll consider (defaults to 100)
    :returns: a network with the given degree distribution'''

    # construct degrees according to the distribution given
    # by the model function
    ns = []
    t = 0
    for i in range(N):
        while True:
            k = 1 + int (np.random.random() * (maxdeg - 1))
            if np.random.random() < p(k):
                ns = ns + [ k ]
                t = t + k
                break

    # if the sequence is odd, choose a random element
    # and increment it by 1 (this doesn't change the
    # distribution significantly, and so is safe)
    if t % 2 != 0:
        i = int(np.random.random() * len(ns))
        ns[i] = ns[i] + 1

    # populate the network using the configuration
    # model with the given degree distribution
    g = nx.configuration_model(ns, create_using = nx.Graph())
    g = g.subgraph(max(nx.connected_components(g), key = len)).copy()
    #g.remove_edges_from(list(g.selfloop_edges()))
    return g


def makePowerlawWithCutoff(alpha, kappa ):
    '''Create a model function for a powerlaw distribution with exponential cutoff.

    :param alpha: the exponent of the distribution
    :param kappa: the degree cutoff
    :returns: a model function'''
    C = Li(alpha, math.exp(-1.0 / kappa))
    def p( k ):
        return (pow((k + 0.0), -alpha) * math.exp(-(k + 0.0) / kappa)) / C
    return p

def make_powerlaw(alpha):
    '''Create a model function for a powerlaw distribution.
    :param alpha: the exponent of the distribution
    :returns: a model function'''
    C = 1.0 / zeta(alpha, 1)
    def p(k):
        return C * pow((k + 0.0), -alpha)
    return p


################################
################################ Plot
################################

def plotSensScaleFree(rc_30, rc_31, rc_32, rc_33,rc_34, rc_35, rc_36, rc_37, rc_38, rc_39, rc_40, totalTS, titletxt):
    totalTS = 1000
    times=[]; I_30=[]; I_31=[]; I_32=[]; I_33=[]; I_34=[]; I_35=[]
    I_36=[]; I_37=[]; I_38=[]; I_39=[]; I_40=[]
    for i in range(1000):
        print(i)
        times.append(rc_30['results']['progress'][i][0])
        I_30.append(rc_30['results']['progress'][i][1]['I']*100/len(g30))
        I_31.append(rc_31['results']['progress'][i][1]['I']*100/len(g31))
        I_32.append(rc_32['results']['progress'][i][1]['I']*100/len(g32))
        #I_33.append(rc_33['results']['progress'][i][1]['I']*100/len(g33))
        #I_34.append(rc_34['results']['progress'][i][1]['I']*100/len(g34))
        #I_35.append(rc_35['results']['progress'][i][1]['I']*100/len(g35))
        # I_36.append(rc_36['results']['progress'][i][1]['I']*100/len(g36))
        # I_37.append(rc_37['results']['progress'][i][1]['I']*100/len(g37))
        # I_38.append(rc_38['results']['progress'][i][1]['I']*100/len(g38))
        # I_39.append(rc_39['results']['progress'][i][1]['I']*100/len(g39))
        # I_40.append(rc_40['results']['progress'][i][1]['I']*100/len(g40))
    plt.plot(times, I_30, label='alpha = 3.0')
    plt.plot(times, I_31, label='alpha = 3.1')
    plt.plot(times, I_32, label='alpha = 3.2')
    #plt.plot(times, I_33, label='alpha = 3.3')
    #plt.plot(times, I_34, label='alpha = 3.4')
    #plt.plot(times, I_35, label='alpha = 3.5')
    # plt.plot(times, I_36, label='alpha = 3.6')
    # plt.plot(times, I_37, label='alpha = 3.7')
    # plt.plot(times, I_38, label='alpha = 3.8')
    # plt.plot(times, I_39, label='alpha = 3.9')
    # plt.plot(times, I_40, label='alpha = 4.0')
    plt.ylabel('fraction de la population [%]')
    plt.xlabel('temps [periodes]')
    plt.title('progression du compartiment I en fonction de l\'exporant α de (3)' )
    plt.legend()
    plt.show()
    
    
def plotSensR0(rc_min, rc_max, rc_median, rc_mean, N, totalTS, titletxt):
    times=[]; I_min=[]; I_max=[]; I_mean=[]; I_median=[];
    for i in range(totalTS):
        times.append(rc_mean['results']['progress'][i][0])
        I_min.append(rc_min['results']['progress'][i][1]['I']*100/N)
        I_max.append(rc_max['results']['progress'][i][1]['I']*100/N)
        I_mean.append(rc_mean['results']['progress'][i][1]['I']*100/N)
        I_median.append(rc_median['results']['progress'][i][1]['I']*100/N)
    plt.plot(times, I_min, label='R0 = 1.4 (min)', color='#F08080')
    plt.plot(times, I_max, label='R0 = 6.49 (max)', color='#00008B')
    plt.plot(times, I_median, label='R0 = 2.79 (median)', color='#A52A2A')
    plt.plot(times, I_mean, label='R0 = 3.28 (mean)', color='red')
    plt.ylabel('fraction de la population [%]')
    plt.xlabel('temps [periodes]')
    plt.title(titletxt)
    plt.legend()
    plt.show()

def plotEvol(rc, N, totalTS, titletxt):
    times=[];S=[]; I=[]; R=[]
    for i in range(totalTS):
        times.append(rc['results']['progress'][i][0])
        S.append(rc['results']['progress'][i][1]['S']*100/N)
        I.append(rc['results']['progress'][i][1]['I']*100/N)
        R.append(rc['results']['progress'][i][1]['R']*100/N)
    plt.plot(times, S, label='Susceptibles', color='orange')
    plt.plot(times, I, label='Infectieux', color='red')
    plt.plot(times, R, label='Rétablis', color='green')
    plt.ylabel('fraction de la population [%]')
    plt.xlabel('temps [periodes]')
    plt.title(titletxt)
    plt.legend()
    plt.show()


def plotPowerLaw():
    x = np.linspace(1, 100, 100)
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))
    y3 = np.zeros(len(x))
    for i in range(len(x)):
        y1[i] = x[i]**(-3)
        y2[i] = x[i]**(-3.5)
        y3[i] = x[i]**(-4)

    plt.plot(x, y1, label='alpha = 3')
    plt.plot(x, y2, label='alpha = 3.5')
    plt.plot(x, y3, label='alpha = 4')
    plt.xscale('log')
    plt.ylabel('Pourcentage de noeuds [%]')
    plt.xlabel('Nombre de liens')
    plt.legend()
    plt.show()
    #plt.rcParams.update({'font.size': 22})
    #plt.savefig('plot1.png', dpi=200)


################################
################################ RUN
################################

def run():
    #Parameters
    N = 15000 #1370000
    g30 = generateFrom(N, make_powerlaw(3))
    g31 = generateFrom(N, make_powerlaw(3.1))
    g32 = generateFrom(N, make_powerlaw(3.2))
    g33 = generateFrom(N, make_powerlaw(3.3))
    g34 = generateFrom(N, make_powerlaw(3.4))
    g35 = generateFrom(N, make_powerlaw(3.5))
    g36 = generateFrom(N, make_powerlaw(3.6))
    g37 = generateFrom(N, make_powerlaw(3.7))
    g38 = generateFrom(N, make_powerlaw(3.8))
    g39 = generateFrom(N, make_powerlaw(3.9))
    g40 = generateFrom(N, make_powerlaw(4))

    param = dict()
    param[ep.SIR.P_INFECT] = 0.0157    # beta .... infection probability
    param[ep.SIR.P_REMOVE] = 0.0048    # gamma ... recovery probability
    param[ep.SIR.P_INFECTED] = 0.01    # initial fraction infected
    param[MonitoredSIR.INTERVAL] = 1

    e = ep.StochasticDynamics(MonitoredSIR(), g33)   # use stochastic (Gillespie) dynamics
    e.process().setMaximumTime(1000)
    rc = e.set(param).run() # set the parameters we want and run the simulation

    #plotEvol(rc, 1122, 1000, 'progression de l\'infection (β=0.0157 et γ=0.0048)')
    param[ep.SIR.P_INFECT] = 0.0157
    rc_mean = e.set(param).run()

    param[ep.SIR.P_INFECT] = 0.0067
    rc_min = e.set(param).run()

    param[ep.SIR.P_INFECT] = 0.03115
    rc_max = e.set(param).run()

    param[ep.SIR.P_INFECT] = 0.01339
    rc_median = e.set(param).run()

    plotSensR0(rc_min, rc_max, rc_median, rc_mean, 1122, 1000, 'progression du compartiment I en fonction de R0')


    #param[ep.SIR.P_INFECT] = 0.0157
    e = ep.StochasticDynamics(MonitoredSIR(), g30)
    e.process().setMaximumTime(1000)
    rc_30 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g31)
    e.process().setMaximumTime(1000)
    rc_31 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g32)
    e.process().setMaximumTime(1000)
    rc_32 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g33)
    e.process().setMaximumTime(1000)
    rc_33 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g34)
    e.process().setMaximumTime(1000)
    rc_34 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g35)
    e.process().setMaximumTime(1000)
    rc_35 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g36)
    e.process().setMaximumTime(1000)
    rc_36 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g37)
    e.process().setMaximumTime(1000)
    rc_37 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g38)
    e.process().setMaximumTime(1000)
    rc_38 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g39)
    e.process().setMaximumTime(1000)
    rc_39 = e.set(param).run()

    e = ep.StochasticDynamics(MonitoredSIR(), g40)
    e.process().setMaximumTime(1000)
    rc_40 = e.set(param).run()

    plotSensScaleFree(rc_30, rc_31, rc_32, rc_33,rc_34, rc_35, rc_36, rc_37, rc_38, rc_39, rc_40, 1000, "titletxt")

run()
    



# SIR class (without using epidemic package)
# class SIR(CompartmentedModel):
#     # the possible dynamics states of a node for SIR dynamics
#     SUSCEPTIBLE = 'S'
#     INFECTED = 'I'
#     REMOVED = 'R'
#
#     # the model parameters
#     P_INFECTED = 'pInfected'
#     P_INFECT = 'pInfect'
#     P_REMOVE = 'pRemove'
#
#     # the edges at which dynamics can occur
#     SI = 'SI'
#
#     def build( self, params ):
#         pInfected = params[self.P_INFECTED]
#         pInfect = params[self.P_INFECT]
#         pRemove = params[self.P_REMOVE]
#         self.addCompartment(self.INFECTED, pInfected)
#         self.addCompartment(self.REMOVED, 0.0)
#         self.addCompartment(self.SUSCEPTIBLE, 1 - pInfected)
#         self.trackNodesInCompartment(self.INFECTED)
#         self.trackEdgesBetweenCompartments(self.SUSCEPTIBLE, self.INFECTED, name=self.SI)
#         self.addEventPerElement(self.SI, pInfect, self.infect)
#         self.addEventPerElement(self.INFECTED, pRemove, self.remove)
#
#     def infect( self, t, e ):
#         (n, m) = e
#         self.changeCompartment(n, self.INFECTED)
#         self.markOccupied(e, t)
#
#     def remove( self, t, n ):
#         self.changeCompartment(n, self.REMOVED)
