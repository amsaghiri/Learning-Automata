import math
import numpy as np
import matplotlib.pyplot as plt

class Automata:
    def __init__(self,r,lambda_1,lambda_2):
         self.r=int(r)
         self.lambda_1=lambda_1
         self.lambda_2=lambda_2
         if(r==0):
            raise ValueError("Must have at least one action.")
         self.probability_list = [1/self.r for x in range(self.r)]
         self.action_selected_list=[]
         self.action_last=-1
         self.entropy_list=[]
         self.favor_action_list=[]

    def decision(self):
        self.action_last=np.random.choice(np.arange(1, r+1),1, self.probability_list)[0]
        self.action_selected_list.append(self.action_last)
        self.favor_action_list.append(self.cal_probablable_action())
        self.entropy_list.append(self.cal_entropy())
        return self.action_last

    def update(self,B):
        for index in range(len(self.probability_list)):
            p=self.probability_list[index]
            if(index==self.action_last-1):
                #Pi(k + 1) = Pi(k) + lambda_1*B(k)(1 - Pi(k)) - lambda_2(1 -B(k))Pi(k)
                self.probability_list[index] = p+(self.lambda_1*B*(1-p))-(self.lambda_2*(1-B)*p)
            else:
                #pj(k) - lambda_1*B(k)pj(k)+lambda_2(1 - B(k))((1/r-1) - Pj(k))
                self.probability_list[index] = p-(self.lambda_1*B*p)+(self.lambda_2*(1-B)*(1/(self.r-1)-p))
    def cal_entropy(self):
        entropy=0
        for index in range(len(self.probability_list)):
            p=self.probability_list[index]
            entropy+=-1*math.log(p)*p
        return entropy
    def cal_probablable_action(self):
        return [i for i, j in enumerate(self.probability_list) if j == max(self.probability_list)][0]+1
def simulated_reward(r,p_environment,action_selected):
    favorable_action=[i for i, j in enumerate(p_environment) if j == max(p_environment)][0]+1
    if action_selected==favorable_action:
        return 1
    return 0
#setting
num_iteration=10000
r=7
lambda_1=0.01
lambda_2=0
favor_env_probability=0.8

p_environment=[]
#set third action as favor action!
for i in range(r):
    if i==2:
        p_environment.append(favor_env_probability)
    else:
        p_environment.append((1-favor_env_probability)/(r-1))

automata_instance=Automata(r,lambda_1,lambda_2)
i=0
B_list=[]
while i<num_iteration:
    action_selected=automata_instance.decision()
    B=simulated_reward(r,p_environment,action_selected)
    B_list.append(B)
    automata_instance.update(B)
    i += 1

#Favor action plot
plt.figure()
plt.plot(np.arange(0,num_iteration),automata_instance.favor_action_list)
plt.title('Favor action plot For r='+str(r))
plt.xlabel('n')
plt.ylabel('Favor action')
plt.savefig('favor_action.png')
#entropy plot
plt.figure()
plt.plot(np.arange(0,num_iteration),automata_instance.entropy_list)
plt.title('Entropy plot For r='+str(r))
plt.xlabel('n')
plt.ylabel('Entropy')
plt.savefig('entropy.png')
#
print("Average Reward="+str(sum(B_list)/len(B_list)))
