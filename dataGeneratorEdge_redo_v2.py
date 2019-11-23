from scipy.integrate import solve_ivp
import numpy as np
from scipy.integrate import odeint
# nodes list is always A -> B  gets represented as [A,B]

class dataGenerator:
    def __init__(self):
        self.initial_conditions = [
                1000, 3500, 0, 100000, 0,
                80000, 0, 50000, 0, 500000,
                0, 50000, 0, 500000, 100000,
                0, 0, 200000, 0, 300000,
                0, 0, 0, 100000, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0
                ]
        self.custom_ranges = [
                [0,2000], [0, 7000], [0, 1], [0, 200000], [0,1],
                [0, 160000], [0, 1], [0, 100000], [0, 1], [0, 1000000],
                [0, 1], [0, 100000], [0, 1], [0, 1000000], [0, 200000],
                [0, 1], [0, 1], [0, 400000], [0, 1], [0, 600000],
                [0, 1], [0, 1], [0, 1], [0, 200000], [0, 1],
                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
                ]
                
        self.custom_ranges2 = [
                [2000,2000*2 ], [7000, 7000*2], [0, 1], [0, 200000, 200000 *2], [0,1],
                [160000, 160000*2], [0, 1], [100000, 100000*2], [0, 1], [1000000, 1000000*2],
                [0, 1], [100000, 100000*2], [0, 1], [1000000, 1000000*2], [200000, 200000*2],
                [0, 1], [0, 1], [400000, 400000*2], [0, 1], [600000, 600000*2],
                [0, 1], [0, 1], [0, 1], [200000, 200000*2], [0, 1],
                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
                ]
        
    def sampleInitialCond(self, sampling_regions): # sampling_regions is a list of lists of 2 elements, each lists of two is a range
        sampled_initial_conditions = []
        for element in sampling_regions:
            sample_space = []
            for i in range(element[0], element[1]):
                sample_space.append(i)
            sample = np.random.choice(np.array(sample_space), 1)[0]
            sampled_initial_conditions.append(sample)
            
        return sampled_initial_conditions
                

    def generateData(self, time, initial_cond):
        res = solve_ivp(self.diffEq, time, initial_cond) 
        return res
    
    def generateDatav2(self, time, initial_cond):
        # time[0] is starting point, time[1] is end pont, time[2] is number of steps
        t = np.linspace(time[0], time[1], time[2])
        sol = odeint(self.diffEqv2, np.array(initial_cond), t)
        
        return sol
        
        
    def diffEqv2(self, x , t):
        k1 = 10**-7
        kr1 = 10**-3
        kc1 =  1 
        
        k2 = 10**-6
        kr2 = 10**-3
        kc2 =  1 
        
        k3 = 3 * 10**-8
        kr3 = 10**-3
        kc3 =  1 
        
        k4 = 10**-7
        kr4 = 10**-3
        kc4 =  1 
        
        k5 = 10**-6
        kr5 = 10**-3
        kc5 =  None
        
        k6 = 2 * 10**-6
        kr6 = 10**-3
        kc6 =  0.01
        
        k7 = 2 * 10**-6
        kr7 = 10**-3
        kc7 =  0.01   
        
        k8 = 10**-4
        kr8 = None
        kc8 =  None
        
        k9 = 5 * 10**-9
        kr9 = 10**-3
        kc9 =  1
        
        k10 = 10**-8
        kr10 = 2 * 10**-4
        kc10 =  1
        
        k11 = 10**-9
        kr11 = 10**-3
        kc11 =  None
        
        k12 = 10**-6
        kr12 = 10**-3
        kc12 =  None
        
        k13 = 3.5 * 10**-6
        kr13 = 10**-3
        kc13 =  None
        
        k14 = 10**-3
        kr14 = 10**-6
        kc14 =  None
        
        k15 = 10**-6
        kr15 = 10**-3
        kc15 =  None
        
        k16 = None
        kr16 = None
        kc16 =  None
        
        k17 = 3.5 * 10**-6
        kr17 = 10**-3
        kc17 =  1
        
        k18 = 10**-9
        kr18 = 10**-3
        kc18 =  1
        
        k19 = 7 * 10**-5
        kr19 = 1.67 * 10**-5
        kc19 =  1.67 * 10**-4
        
        k20 = 10**-8
        kr20 = 1.67 * 10**-3
        kc20 =  1.67 * 10**-2
        
        k21 = 0.01
        kr21 = 0.01
        kc21 =  None
        
        k22 = 10**-7
        kr22 = 10**-3
        kc22 =  1
        
        k23 = 2 * 10**-6
        kr23 = 10**-3
        kc23 =  0.01
        
        s25To41 = 0
        ri_allbut_5 = 5.79 * 10**-6
        r5 = 2.89 * 10**-5
        
        s1 = 0
        s2 = 2.03*10**-2
        
        s3 = 0
        s4 = 5.79*10**-1
        
        s5 = 0
        s6 = 4.63*10**-1
        
        s7 = 0
        s8 = 2.89*10**-1
        
        s9 = 0
        s10 = 2.89
        
        s11 = 0
        s12 = 2.89*10**-1
        
        s13 = 0
        s14 = 2.89
        
        s15 = 5.79 * 10**-1
        s16 = 0
        
        s17 = 0
        s18 = 1.16*10**-1
        
        s19 = 0
        s20 = 1.74*10**-1
        
        s21 = 0
        s22 = 0
        
        s23 = 0
        s24 = 5.79*10**-1
        
        v = 0.07
        
        ep1 = -k1*x[0]*x[1] + kr1*x[25] + s1 - ri_allbut_5*x[0]
        
        ep2 = -k1*x[0]*x[1] + kr1*x[25] - k3*x[4]*x[2] + kr3*x[27] + s2 - ri_allbut_5*x[1] 
        
        ep3 = kc1*x[25] -k2*x[2]*x[3] +kr2*x[26] +kc2*x[26] +kc3*x[27] -k4*x[2]*x[5] +kr4*x[28] +kc4*x[28] -ri_allbut_5*x[2]
        
        ep4 = -k2*x[2]*x[3] +kr2*x[26] -k9*x[10]*x[3] +kr9*x[31] +s4 -ri_allbut_5*x[3]
        
        ep5 = kc2*x[26] -k3*x[4]*x[1] +kr3*x[27] +kc3*x[27] +kc9*x[31] -k19*x[4]*x[19] +kr19*x[36] -k22*x[4]*x[23] +kr22*x[34] +kc22*x[34] -r5*x[4]
        
        ep6 = -k4*x[2]*x[5] +kr4*x[28] +s6 -ri_allbut_5*x[5]
        
        ep7 = kc4*x[28] -k15*x[15]*x[6] +kr15*x[22] -k21*x[6] +kr21*x[21] -ri_allbut_5*x[6]
        
        ep8 = -(1/v**2)* k5*x[21]*x[7] +kr5*x[8] +k8*x[8] +s8 -ri_allbut_5*x[7]
        
        ep9 = (1/v**2)*k5*x[21]*x[7] -kr5*x[8] -(1/v**2)*k6*x[8]*x[9] +kr6*x[29] +kc6*x[29] -(1/v**2)*k7*x[8]*x[11] + kr7*x[30] +kc6*x[30] -k8*x[8]
        
        ep10 = -(1/v**2)*k6*x[8]*x[9] +kr6*x[29] +s10 -ri_allbut_5*x[9]
        
        ep11 = kc6*x[29] -k9*x[10]*x[3] +kr9*x[31] +kc9*x[31] -k11*x[15]*x[10] +kr11*x[16] -k23*x[19]*x[10] +kr23*x[40] -ri_allbut_5*x[10]
        
        ep12 = -(1/v**2)*k7*x[8]*x[11] +kr7*x[30] +s12 -ri_allbut_5*x[11]
        
        ep13 = kc7*x[30] -k13*x[12]*x[19] +kr13*x[20] -k17*x[12]*x[19]*x[15] +kr17*x[33] -ri_allbut_5*x[12]
        
        ep14 = -k10*x[13]*x[15] +kr10*x[32] +kc10*x[32] -k20*x[13]*x[19] +kr20*x[38] +s14 -ri_allbut_5*x[13]
        
        ep15 = kc10*x[32] -k14*x[14] +kr14*x[15] +kc17*x[33] +k18*x[35] +s15 -ri_allbut_5*x[14]
        
        ep16 = -k10*x[13]*x[15] +kr10*x[32] -k11*x[15]*x[10] +kr11*x[16] +k14*x[14] -kr14*x[15] -k15*x[15]*x[6] +kr15*x[22] -k17*x[12]*x[19]*x[15] +kr17*x[33] -k18*x[19]*x[15] +kr18*x[35] -ri_allbut_5*x[15]
        
        ep17 = k11*x[15]*x[10] -kr11*x[16] -ri_allbut_5*x[16]
        
        ep18 = -(1/v**2)*k12*x[17]*x[21] +kr12*x[18] +s18 -ri_allbut_5*x[17]
        
        ep19 = (1/v**2)*k12*x[17]*x[21] -kr12*x[18] -ri_allbut_5*x[18]
        
        ep20 = -k13*x[12]*x[19] +kr13*x[20] -k17*x[12]*x[19]*x[15] +kr17*x[33] -k18*x[15]*x[19] +kr18*x[35]+kc18*x[35] -k19*x[4]*x[19] +kr19*x[36] +kc19*x[36] -k20*x[13]*x[19] +kr20*x[38] +kc20*x[38] -k23*x[19]*x[10]+kr23*x[40] +s20 -ri_allbut_5*x[19]*(1/(x[15]+1))
        
        ep21 = k13*x[12]*x[19] -kr13*x[20] +kc17*x[33] -ri_allbut_5*x[20]
        
        ep22 = -(1/v**2)*k5*x[21]*x[7] +kr5*x[8] -(1/v**2)*k12*x[17]*x[21] +kr12*x[18] +k21*x[6] -kr21*x[21] -ri_allbut_5*x[21] 
        
        ep23 = k15*x[15]*x[6] -kr15*x[22] -ri_allbut_5*x[22]
        
        ep24 = -k22*x[4]*x[23] +kr22*x[34] +s24 -ri_allbut_5*x[23]
        
        ep25 = kc22*x[34] -ri_allbut_5*x[24]
        
        ep26 = k1*x[0]*x[1] -kr1*x[25] -kc1*x[25] -ri_allbut_5*x[25]
        
        ep27 = k2*x[2]*x[3] -kr2*x[26] -kc2*x[26] -ri_allbut_5*x[26]
        
        ep28 = k3*x[4]*x[1] -kr3*x[27] -kc3*x[27] -ri_allbut_5*x[27]
        
        ep29 = k4*x[2]*x[5] -kr4*x[28] -kc4*x[28] -ri_allbut_5*x[28]
    
        ep30 = (1/v**2)*k6*x[8]*x[9] -kr6*x[29] -kc6*x[29] -ri_allbut_5*x[29]
        
        ep31 = (1/v**2)*k7*x[8]*x[11] -kr7*x[30] -kc6*x[30] -ri_allbut_5*x[30]
        
        ep32 = k9*x[10]*x[3] -kr9*x[31] -kc9*x[31] -ri_allbut_5*x[31]
        
        ep33 = k10*x[13]*x[15] -kr10*x[32] -kc10*x[32] -ri_allbut_5*x[32]
        
        ep34 = k17*x[12]*x[19]*x[15] -kr17*x[33] -kc17*x[33] -ri_allbut_5*x[33]
        
        ep35 = k22*x[4]*x[23] -kr22*x[34] -kc22*x[34] -ri_allbut_5*x[34]
        
        ep36 = k18*x[15]*x[19] -kr18*x[35] -kc18*x[35] -ri_allbut_5*x[35]
        
        ep37 = k19*x[4]*x[19] -kr19*x[36] -kc19*x[36] -ri_allbut_5*x[36]
        
        ep38 = kc19*x[36] -ri_allbut_5*x[37] 
        
        ep39 = k20*x[13]*x[19] -kr20*x[38] -kc20*x[38] -ri_allbut_5*x[38] 
        
        ep40 = kc20*x[38] -ri_allbut_5*x[39]
        
        ep41 = k23*x[19]*x[10] -kr23*x[40] -ri_allbut_5*x[40]
        
        
        return [ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, 
                ep11, ep12, ep13, ep14, ep15, ep16, ep17, ep18, ep19, ep20, 
                ep21, ep22, ep23, ep24, ep25, ep26, ep27, ep28, ep29, ep30, 
                ep31, ep32, ep33, ep34, ep35, ep36, ep37, ep38, ep39, ep40, 
                ep41]

    def diffEqFixedPoint(self, x , t, nodeIndex, nodeVal = 0):
        k1 = 10**-7
        kr1 = 10**-3
        kc1 =  1 
        
        k2 = 10**-6
        kr2 = 10**-3
        kc2 =  1 
        
        k3 = 3 * 10**-8
        kr3 = 10**-3
        kc3 =  1 
        
        k4 = 10**-7
        kr4 = 10**-3
        kc4 =  1 
        
        k5 = 10**-6
        kr5 = 10**-3
        kc5 =  None
        
        k6 = 2 * 10**-6
        kr6 = 10**-3
        kc6 =  0.01
        
        k7 = 2 * 10**-6
        kr7 = 10**-3
        kc7 =  0.01   
        
        k8 = 10**-4
        kr8 = None
        kc8 =  None
        
        k9 = 5 * 10**-9
        kr9 = 10**-3
        kc9 =  1
        
        k10 = 10**-8
        kr10 = 2 * 10**-4
        kc10 =  1
        
        k11 = 10**-9
        kr11 = 10**-3
        kc11 =  None
        
        k12 = 10**-6
        kr12 = 10**-3
        kc12 =  None
        
        k13 = 3.5 * 10**-6
        kr13 = 10**-3
        kc13 =  None
        
        k14 = 10**-3
        kr14 = 10**-6
        kc14 =  None
        
        k15 = 10**-6
        kr15 = 10**-3
        kc15 =  None
        
        k16 = None
        kr16 = None
        kc16 =  None
        
        k17 = 3.5 * 10**-6
        kr17 = 10**-3
        kc17 =  1
        
        k18 = 10**-9
        kr18 = 10**-3
        kc18 =  1
        
        k19 = 7 * 10**-5
        kr19 = 1.67 * 10**-5
        kc19 =  1.67 * 10**-4
        
        k20 = 10**-8
        kr20 = 1.67 * 10**-3
        kc20 =  1.67 * 10**-2
        
        k21 = 0.01
        kr21 = 0.01
        kc21 =  None
        
        k22 = 10**-7
        kr22 = 10**-3
        kc22 =  1
        
        k23 = 2 * 10**-6
        kr23 = 10**-3
        kc23 =  0.01
        
        s25To41 = 0
        ri_allbut_5 = 5.79 * 10**-6
        r5 = 2.89 * 10**-5
        
        s1 = 0
        s2 = 2.03*10**-2
        
        s3 = 0
        s4 = 5.79*10**-1
        
        s5 = 0
        s6 = 4.63*10**-1
        
        s7 = 0
        s8 = 2.89*10**-1
        
        s9 = 0
        s10 = 2.89
        
        s11 = 0
        s12 = 2.89*10**-1
        
        s13 = 0
        s14 = 2.89
        
        s15 = 5.79 * 10**-1
        s16 = 0
        
        s17 = 0
        s18 = 1.16*10**-1
        
        s19 = 0
        s20 = 1.74*10**-1
        
        s21 = 0
        s22 = 0
        
        s23 = 0
        s24 = 5.79*10**-1
        
        v = 0.07
        
        ep1 = -k1*x[0]*x[1] + kr1*x[25] + s1 - ri_allbut_5*x[0]
        
        ep2 = -k1*x[0]*x[1] + kr1*x[25] - k3*x[4]*x[2] + kr3*x[27] + s2 - ri_allbut_5*x[1] 
        
        ep3 = kc1*x[25] -k2*x[2]*x[3] +kr2*x[26] +kc2*x[26] +kc3*x[27] -k4*x[2]*x[5] +kr4*x[28] +kc4*x[28] -ri_allbut_5*x[2]
        
        ep4 = -k2*x[2]*x[3] +kr2*x[26] -k9*x[10]*x[3] +kr9*x[31] +s4 -ri_allbut_5*x[3]
        
        ep5 = kc2*x[26] -k3*x[4]*x[1] +kr3*x[27] +kc3*x[27] +kc9*x[31] -k19*x[4]*x[19] +kr19*x[36] -k22*x[4]*x[23] +kr22*x[34] +kc22*x[34] -r5*x[4]
        
        ep6 = -k4*x[2]*x[5] +kr4*x[28] +s6 -ri_allbut_5*x[5]
        
        ep7 = kc4*x[28] -k15*x[15]*x[6] +kr15*x[22] -k21*x[6] +kr21*x[21] -ri_allbut_5*x[6]
        
        ep8 = -(1/v**2)* k5*x[21]*x[7] +kr5*x[8] +k8*x[8] +s8 -ri_allbut_5*x[7]
        
        ep9 = (1/v**2)*k5*x[21]*x[7] -kr5*x[8] -(1/v**2)*k6*x[8]*x[9] +kr6*x[29] +kc6*x[29] -(1/v**2)*k7*x[8]*x[11] + kr7*x[30] +kc6*x[30] -k8*x[8]
        
        ep10 = -(1/v**2)*k6*x[8]*x[9] +kr6*x[29] +s10 -ri_allbut_5*x[9]
        
        ep11 = kc6*x[29] -k9*x[10]*x[3] +kr9*x[31] +kc9*x[31] -k11*x[15]*x[10] +kr11*x[16] -k23*x[19]*x[10] +kr23*x[40] -ri_allbut_5*x[10]
        
        ep12 = -(1/v**2)*k7*x[8]*x[11] +kr7*x[30] +s12 -ri_allbut_5*x[11]
        
        ep13 = kc7*x[30] -k13*x[12]*x[19] +kr13*x[20] -k17*x[12]*x[19]*x[15] +kr17*x[33] -ri_allbut_5*x[12]
        
        ep14 = -k10*x[13]*x[15] +kr10*x[32] +kc10*x[32] -k20*x[13]*x[19] +kr20*x[38] +s14 -ri_allbut_5*x[13]
        
        ep15 = kc10*x[32] -k14*x[14] +kr14*x[15] +kc17*x[33] +k18*x[35] +s15 -ri_allbut_5*x[14]
        
        ep16 = -k10*x[13]*x[15] +kr10*x[32] -k11*x[15]*x[10] +kr11*x[16] +k14*x[14] -kr14*x[15] -k15*x[15]*x[6] +kr15*x[22] -k17*x[12]*x[19]*x[15] +kr17*x[33] -k18*x[19]*x[15] +kr18*x[35] -ri_allbut_5*x[15]
        
        ep17 = k11*x[15]*x[10] -kr11*x[16] -ri_allbut_5*x[16]
        
        ep18 = -(1/v**2)*k12*x[17]*x[21] +kr12*x[18] +s18 -ri_allbut_5*x[17]
        
        ep19 = (1/v**2)*k12*x[17]*x[21] -kr12*x[18] -ri_allbut_5*x[18]
        
        ep20 = -k13*x[12]*x[19] +kr13*x[20] -k17*x[12]*x[19]*x[15] +kr17*x[33] -k18*x[15]*x[19] +kr18*x[35]+kc18*x[35] -k19*x[4]*x[19] +kr19*x[36] +kc19*x[36] -k20*x[13]*x[19] +kr20*x[38] +kc20*x[38] -k23*x[19]*x[10]+kr23*x[40] +s20 -ri_allbut_5*x[19]*(1/(x[15]+1))
        
        ep21 = k13*x[12]*x[19] -kr13*x[20] +kc17*x[33] -ri_allbut_5*x[20]
        
        ep22 = -(1/v**2)*k5*x[21]*x[7] +kr5*x[8] -(1/v**2)*k12*x[17]*x[21] +kr12*x[18] +k21*x[6] -kr21*x[21] -ri_allbut_5*x[21] 
        
        ep23 = k15*x[15]*x[6] -kr15*x[22] -ri_allbut_5*x[22]
        
        ep24 = -k22*x[4]*x[23] +kr22*x[34] +s24 -ri_allbut_5*x[23]
        
        ep25 = kc22*x[34] -ri_allbut_5*x[24]
        
        ep26 = k1*x[0]*x[1] -kr1*x[25] -kc1*x[25] -ri_allbut_5*x[25]
        
        ep27 = k2*x[2]*x[3] -kr2*x[26] -kc2*x[26] -ri_allbut_5*x[26]
        
        ep28 = k3*x[4]*x[1] -kr3*x[27] -kc3*x[27] -ri_allbut_5*x[27]
        
        ep29 = k4*x[2]*x[5] -kr4*x[28] -kc4*x[28] -ri_allbut_5*x[28]
    
        ep30 = (1/v**2)*k6*x[8]*x[9] -kr6*x[29] -kc6*x[29] -ri_allbut_5*x[29]
        
        ep31 = (1/v**2)*k7*x[8]*x[11] -kr7*x[30] -kc6*x[30] -ri_allbut_5*x[30]
        
        ep32 = k9*x[10]*x[3] -kr9*x[31] -kc9*x[31] -ri_allbut_5*x[31]
        
        ep33 = k10*x[13]*x[15] -kr10*x[32] -kc10*x[32] -ri_allbut_5*x[32]
        
        ep34 = k17*x[12]*x[19]*x[15] -kr17*x[33] -kc17*x[33] -ri_allbut_5*x[33]
        
        ep35 = k22*x[4]*x[23] -kr22*x[34] -kc22*x[34] -ri_allbut_5*x[34]
        
        ep36 = k18*x[15]*x[19] -kr18*x[35] -kc18*x[35] -ri_allbut_5*x[35]
        
        ep37 = k19*x[4]*x[19] -kr19*x[36] -kc19*x[36] -ri_allbut_5*x[36]
        
        ep38 = kc19*x[36] -ri_allbut_5*x[37] 
        
        ep39 = k20*x[13]*x[19] -kr20*x[38] -kc20*x[38] -ri_allbut_5*x[38] 
        
        ep40 = kc20*x[38] -ri_allbut_5*x[39]
        
        ep41 = k23*x[19]*x[10] -kr23*x[40] -ri_allbut_5*x[40]
        
        sol = [ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, 
                ep11, ep12, ep13, ep14, ep15, ep16, ep17, ep18, ep19, ep20, 
                ep21, ep22, ep23, ep24, ep25, ep26, ep27, ep28, ep29, ep30, 
                ep31, ep32, ep33, ep34, ep35, ep36, ep37, ep38, ep39, ep40, 
                ep41]
        
        sol[nodeIndex] = nodeVal
        
        
        return sol  
        
    def variableList(self):
        l = [
             'TRAIL',
             'pC8',
             'C8*',
             'pC3',
             'C3*',
             'BAX',
             'BAX*',
             'Mito',
             'Mito*',
             'Cytocm',
             'Cytocc',
             'SMACm',
             'SMACc',
             'PTEN',
             'AKT',
             'AKT*',
             'AKT:Cytocc',
             'Bcl-2',
             'Bcl-2:BAXm',
             'XIAP',
             'SMAC:XIAP',
             'BAXm',
             'AKT:BAX*',
             'PARP',
             'cPARP',
             'TRAIL:pC8',
             'C8*:pC3',
             'C3*:pC8',
             'C8*:BAX',
             'Mito:Cytocm',
             'Mito:SMACm',
             'Cytocc:pC3',
             'PTEN:AKT*',
             'SMACc:XIAP:AKT*',
             'C3*:PARP',
             'AKT*:XIAP',
             'C3*:XIAP',
             'C3Ub',
             'PTEN:XIAP',
             'PTENUb',
             'XIAP:Cytocc'
             
             ]
        return l
        
    def generateGraph(self):
        
        eq1Edge = [
                ['pC8','TRAIL'],
                ['TRAIL:pC8', 'TRAIL']
                ]
        eq2Edge = [
                ['TRAIL', 'pC8'],
                ['C3*', 'pC8'],
                ['TRAIL:pC8', 'pC8'],
                ['C3*:pC8', 'pC8']
                ]
        eq3Edge = [
                ['TRAIL:pC8','C8*'],
                ['pC3', 'C8*'],
                ['C8*:pC3', 'C8*'],
                ['C3*:pC8', 'C8*'],
                ['BAX', 'C8*'],
                ['C8*:BAX', 'C8*']
                
                ]
        eq4Edge = [
                ['C8*', 'pC3'],
                ['C8*:pC3', 'pC3'],
                ['Cytocc', 'pC3'],
                ['Cytocc:pC3', 'pC3']
                ]
        eq5Edge = [
                ['C8*:pC3', 'C3*'],
                ['pC8', 'C3*'],
                ['C3*:pC8', 'C3*'],
                ['Cytocc:pC3', 'C3*'],
                ['XIAP', 'C3*'],
                ['C3*:XIAP', 'C3*'],
                ['PARP', 'C3*'],
                ['C3*:PARP', 'C3*']
                ]
        eq6Edge = [
                ['C8*', 'BAX'],
                ['C8*:BAX', 'BAX']
                ]
        eq7Edge = [
                ['C8*:BAX', 'BAX*'],
                ['AKT*', 'BAX*'],
                ['AKT:BAX*', 'BAX*'],
                ['BAXm', 'BAX*']
                
                ]
        eq8Edge =[
                ['BAXm', 'Mito'],
                ['Mito*', 'Mito'],
                
                ]
        eq9Edge = [
                ['BAXm', 'Mito*'],
                ['Mito', 'Mito*'],
                ['Cytocm', 'Mito*'],
                ['Mito:Cytocm', 'Mito*'],
                ['SMACm', 'Mito*'],
                ['Mito:SMACm', 'Mito*']
                                
                ]
        
        eq10Edge = [
                ['Mito*', 'Cytocm'],
                ['Mito:Cytocm', 'Cytocm']
                
                ]
        
        eq11Edge =[
                ['Mito:Cytocm', 'Cytocc'],
                ['pC3', 'Cytocc'],
                ['Cytocc:pC3', 'Cytocc'],
                ['AKT*', 'Cytocc'],
                ['AKT:Cytocc', 'Cytocc'],
                ['XIAP', 'Cytocc'],
                ['XIAP:Cytocc', 'Cytocc']
                                                
                ]
        eq12Edge = [
                ['Mito*', 'SMACm'],
                ['Mito:SMACm', 'SMACm']
                ]
        eq13Edge = [
                ['Mito:SMACm', 'SMACc'],
                ['XIAP', 'SMACc'],
                ['SMAC:XIAP', 'SMACc'],
                ['AKT*', 'SMACc'],
                ['SMACc:XIAP:AKT*', 'SMACc']
                
                ]
        eq14Edge = [
                ['AKT*', 'PTEN'],
                ['PTEN:AKT*', 'PTEN'],
                ['XIAP', 'PTEN'],
                ['PTEN:XIAP', 'PTEN']
                ]
        
        eq15Edge = [
                ['PTEN:AKT*', 'AKT'],
                ['AKT*', 'AKT'],
                ['SMACc:XIAP:AKT*', 'AKT'],
                ['AKT*:XIAP', 'AKT']
                
                ]
        eq16Edge = [
                ['PTEN', 'AKT*'],
                ['PTEN:AKT*', 'AKT*'],
                ['Cytocc', 'AKT*'],
                ['AKT:Cytocc', 'AKT*'],
                ['AKT', 'AKT*'],
                ['BAX*', 'AKT*'],
                ['AKT:BAX*', 'AKT*'],
                ['SMACc', 'AKT*'],
                ['XIAP', 'AKT*'],
                ['SMACc:XIAP:AKT*', 'AKT*'],
                ['AKT*:XIAP', 'AKT*']
                ]
        eq17Edge = [
                ['AKT*', 'AKT:Cytocc'],
                ['Cytocc', 'AKT:Cytocc']
                ]
        eq18Edge = [
                ['BAXm', 'Bcl-2'],
                ['Bcl-2:BAXm', 'Bcl-2']
                ]
        
        eq19Edge = [
                ['Bcl-2', 'Bcl-2:BAXm'],
                ['BAXm', 'Bcl-2:BAXm']
                ]
        
        eq20Edge = [
                ['SMACc', 'XIAP'],
                ['SMAC:XIAP', 'XIAP'],
                ['AKT*', 'XIAP'],
                ['SMACc:XIAP:AKT*', 'XIAP'],
                ['AKT*:XIAP', 'XIAP'],
                ['C3*', 'XIAP'],
                ['C3*:XIAP', 'XIAP'],
                ['PTEN', 'XIAP'],
                ['PTEN:XIAP', 'XIAP'],
                ['Cytocc', 'XIAP'],
                ['XIAP:Cytocc', 'XIAP'],
                ['AKT*', 'XIAP']
                ]
        
        eq21Edge = [
                ['SMACc', 'SMAC:XIAP'],
                ['XIAP', 'SMAC:XIAP'],
                ['SMACc:XIAP:AKT*', 'SMAC:XIAP']
                ]
        
        eq22Edge = [
                ['Mito', 'BAXm'],
                ['Mito*', 'BAXm'],
                ['Bcl-2', 'BAXm'],
                ['Bcl-2:BAXm', 'BAXm'],
                ['BAX*', 'BAXm']
                
                
                ]
        eq23Edge = [
                ['AKT*', 'AKT:BAX*'],
                ['BAX*', 'AKT:BAX*']
                ]
        eq24Edge = [
                ['C3*', 'PARP'],
                ['C3*:PARP', 'PARP']
                ]
        eq25Edge = [
                ['C3*:PARP', 'cPARP']
                ]
        eq26Edge = [
                ['TRAIL', 'TRAIL:pC8'],
                ['pC8', 'TRAIL:pC8']
                ]
        eq27Edge = [
                ['C8*', 'C8*:pC3'],
                ['pC3', 'C8*:pC3']
                ]
        
        eq28Edge = [
                ['C3*', 'C3*:pC8'],
                ['pC8', 'C3*:pC8']
                
                ]
        eq29Edge = [
                ['C8*', 'C8*:BAX'],
                ['BAX', 'C8*:BAX']
                ]
        eq30Edge = [
                ['Mito*', 'Mito:Cytocm'],
                ['Cytocm', 'Mito:Cytocm']
                ]
        eq31Edge = [
                ['Mito*', 'Mito:SMACm'],
                ['SMACm', 'Mito:SMACm']
                ]
        eq32Edge = [
                ['pC3', 'Cytocc:pC3'],
                ['Cytocc', 'Cytocc:pC3']
                ]
        eq33Edge = [
                ['PTEN', 'PTEN:AKT*'],
                ['AKT*', 'PTEN:AKT*']
                ]
        eq34Edge = [
                ['SMACc', 'SMACc:XIAP:AKT*'],
                ['XIAP', 'SMACc:XIAP:AKT*'],
                ['AKT*', 'SMACc:XIAP:AKT*']
                ]
        eq35Edge = [
                ['C3*', 'C3*:PARP'],
                ['PARP', 'C3*:PARP']
                ]
        eq36Edge = [
                ['AKT*', 'AKT*:XIAP'],
                ['XIAP', 'AKT*:XIAP']
                ]
        eq37Edge = [
                ['C3*', 'C3*:XIAP'],
                ['XIAP', 'C3*:XIAP']
                ]
        eq38Edge = [
                ['C3*:XIAP', 'C3Ub']
                ]
        eq39Edge = [
                ['PTEN', 'PTEN:XIAP'],
                ['XIAP', 'PTEN:XIAP']
                
                ]
        eq40Edge = [
                ['PTEN:XIAP', 'PTENUb']
                
                ]
        eq41Edge = [
                ['XIAP', 'XIAP:Cytocc'],
                ['Cytocc', 'XIAP:Cytocc']
                ]
        return eq1Edge + eq2Edge + eq3Edge + eq4Edge + eq5Edge + eq6Edge + eq7Edge + eq8Edge + eq9Edge + eq10Edge + eq11Edge + eq12Edge + eq13Edge + eq14Edge + eq15Edge + eq16Edge + eq17Edge + eq18Edge + eq19Edge + eq20Edge + eq21Edge + eq22Edge + eq23Edge + eq24Edge + eq25Edge + eq26Edge + eq27Edge + eq28Edge + eq29Edge + eq30Edge + eq31Edge + eq32Edge + eq33Edge + eq34Edge + eq35Edge + eq36Edge + eq37Edge + eq38Edge + eq39Edge + eq40Edge + eq41Edge   
    
    
    
    
    def edgeListToNodeDic(self, edgeList): # tested done
        NodeDict = dict()
        
        for edge in edgeList:
            A = edge[0]
            B = edge[1]
            if A == B:
                next
            
            if B in NodeDict.keys():
                NodeDict[B].append(A)
            else:
                NodeDict[B] = [A]
        return NodeDict
    
    def hiddenAndObservableStates(self):
        observables = [
             'TRAIL',
             'pC8',
             'pC3',
             'BAX',
             'Mito',
             'Cytocm',
             'Cytocc',
             'SMACm',
             'SMACc',
             'PTEN',
             'AKT',
             'Bcl-2',
             'XIAP',
             'PARP'
             
             ]
        observables = set(self.variableList())
        
        allStates = set(self.variableList())
        
        hiddenNodes = allStates- observables
        
        return list(hiddenNodes), list(observables)
        
    def backTrackObservable(self, node, nodeDic, hiddenNodes, visitedNodes = [] ):
        #needs to be tested 
        if node not in nodeDic.keys():
            return [node], []
        
        
        inNeighbours = list(set(nodeDic[node]))
        
        if node in hiddenNodes and visitedNodes == []:
            #print('wtf')
            visitedNodes.append(node)
        
        retList = []
        #print(node)
        for inNode in inNeighbours:
            if inNode in visitedNodes:
                #print(inNode)
                #print(node)
                #print(visitedNodes)
                #print("There is a hidden node loop")
                continue
                #exit()
            if inNode in hiddenNodes:
                #print('shit')
                visitedNodes.append(inNode)
                temp, tempVisited = self.backTrackObservable(inNode, nodeDic, hiddenNodes, visitedNodes)
                retList += temp
                
                visitedNodes+= tempVisited
                
                visitedNodes = list(set(visitedNodes))
                
            else:
                retList.append(inNode)
        return list(set(retList)), list(set(visitedNodes))
    
    def graphWithHiddenStates(self, observableNodes, hiddenNodes):
        # returns the node dictionary with 
        edgeList = self.generateGraph()
        nodeDic = self.edgeListToNodeDic(edgeList)
        
        nodeDicObservablesOnly = dict()
        
        for node in observableNodes:
            inNeighbours = nodeDic[node]
            
            obs_neighbours = []
            for neighbour in inNeighbours:
                if neighbour in hiddenNodes:
                    # BackTrack
                    neighbourOfObservables, visitedNodes = self.backTrackObservable(neighbour, nodeDic, hiddenNodes, [])
                    obs_neighbours += neighbourOfObservables
                else: # this neighbour is observable, append to list
                    obs_neighbours.append(neighbour)
                
            nodeDicObservablesOnly[node] = list(set(obs_neighbours) - set([node]))
            
            
        return nodeDicObservablesOnly
    
    
    def edgeListToNodeDicOutNeighbours(self, edgeList): # tested done
        NodeDict = dict()
        
        for edge in edgeList:
            A = edge[0]
            B = edge[1]
            if A == B:
                next
            
            if A in NodeDict.keys():
                NodeDict[A].append(B)
            else:
                NodeDict[A] = [B]
        return NodeDict
    
    def reverseGraph(self, G):
        edgeList = []
        for node in G.keys():
            for element in G[node]:
                edgeList.append([node, element])
                
        return self.edgeListToNodeDic(edgeList)
    

    '''
    def reverseGraphWithHiddenStates(self, observableNodes, hiddenNodes):
        # returns the node dictionary with 
        edgeList = self.generateGraph()
        nodeDic = self.edgeListToNodeDicOutNeighbours(edgeList)
        
        nodeDicObservablesOnly = dict()
        
        for node in observableNodes:
            inNeighbours = nodeDic[node]
            
            obs_neighbours = []
            for neighbour in inNeighbours:
                if neighbour in hiddenNodes:
                    # BackTrack
                    neighbourOfObservables, visitedNodes = self.backTrackObservable(neighbour, nodeDic, hiddenNodes, [])
                    obs_neighbours += neighbourOfObservables
                else: # this neighbour is observable, append to list
                    obs_neighbours.append(neighbour)
                
            nodeDicObservablesOnly[node] = list(set(obs_neighbours) - set([node]))
            
            
        return nodeDicObservablesOnly
        '''
    
    def timeEvolutionAfterPerturbation(self, node, currentState, time = [0, 24*3600, 100*24*3600]):
        t = np.linspace(time[0], time[1], time[2])
        sol = odeint(self.diffEqv2, currentState[self.variableList()].values[0], t)
        
        return sol

    def timeEvolutionAfterPerturbationFixedPoint(self, node, currentState, time = [0, 24*3600, 100*24*3600], nodeVal = 0):
        t = np.linspace(time[0], time[1], time[2])
        nodeIndex = currentState.columns.get_loc(node)
        sol = odeint(self.diffEqFixedPoint, currentState[self.variableList()].values[0], t, (nodeIndex, nodeVal))
        
        return sol


    def timeEvolutionAfterPerturbation2(self, node, currentState, time = [0, 24*3600, 100*24*3600], gap = 10, nodeVal = 0):
        t = np.linspace(time[0], time[1]/3600, time[2]/3600)
        sol = currentState[self.variableList()].values[0]
        index = currentState.columns.get_loc(node)
        print(sol)
        sol[index] = nodeVal
        for i in range(0,3600):
            sol = odeint(self.diffEqv2, sol, t)
            sol = sol[-1,:]
            #print(sol)
            #print(sol.shape)
            #print(index)
            #sol[index] = nodeVal
        return sol








