## Group Members: Sri Santhosh Hari, Kunal Kotian, Devesh Maheshwari, Vinay Patlolla, Jade Yun

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Functions:

def UserSim(n, parameter):
    """
    input:
    n - the number to of users to simulate
    parameter - the rate parameter (lambda)
    
    output: 
    A list of exponential random variable simulations - specifically, the prbability density for each
    simulated random variable instance.
    """
    return list(np.random.exponential(1.0 / parameter, n)) #1st parameter is scale = 1/parameter given aka lambda

def get_survivors(user_pdf, max_time=3, time_step=0.25):
    """
    input:
    user_pdf - probability distribution of an exponential R.V. 
    max_time - max. time
    time_step - time step to create bins
    output:
    a tuple of cumulative survivor counts and corresponding times
    """
    num_bins = int(max_time / time_step)
    counts, bin_edges = np.histogram(user_pdf, bins=num_bins, range=(0, 3), density=False)  # bin the data
    cdf_user_counts = np.cumsum(counts)  # generate cumulative counts of users (similar to cdf)
    survivors = len(user_pdf) - cdf_user_counts      # users remaining at each time instance
    return (list(survivors), list(bin_edges[1:]))

def EstLam1(quitting_time):
    """
    input:
    quitting_time - simulated quitting times of users
    output:
    estimated lambda
    """
    return 1.0/np.mean(np.array(quitting_time))

def bootstrap(n, quitting_time, alpha):
    """
    input: 
    n - number of bootstaps 
    quitting_time - user quitting time acquired from UserSim
    alpha - significance level 
    
    output: 
    lower and upper bound of estimated lambda with respect to given alpha 
    """
    # matrix contains 500 lists of simulated user quitting time
    quitting_times = np.random.choice(quitting_time, size=(n,len(quitting_time)))
    # estimated lamda for each user quiitng time list
    estimated_lambda = np.apply_along_axis(EstLam1, 1, quitting_times)
    
    # upper bound of estimated lambda 
    upper = np.percentile(estimated_lambda, (1.0 -alpha/2)*100)
    # lower bound of estimated lambda 
    lower = np.percentile(estimated_lambda, (alpha/2.0)*100) 
    
    return lower, upper

def HurdleFun(user_quit_times, breakpoints):
    '''
    user_quit_times: list of times at which user quit
    breakpoints: list of breakpoints
    '''
    user_quit_times = np.sort(user_quit_times)
    total_users = user_quit_times.size
    total_quit_prev = 0
    user_quit_bp = list()
    
    for bp in breakpoints:
        # Get the total users who quit so far
        total_quit = user_quit_times[user_quit_times < bp].size
        # Subtract the total users who quit till previous breakpoint to get users who quit at current breakpoint
        user_quit_bp.append(total_quit - total_quit_prev)
        # Keep track of users who quit so far
        total_quit_prev = total_quit
    
    #Lastly add the remaining users who didn
    remaining_users = total_users - total_quit_prev
    user_quit_bp.append(remaining_users)
    return user_quit_bp

def cdf(lam, x):
    '''
    Returns exponential distribution's cdf when lambda and x are given
    '''
    return (1 - np.exp(-1*x*lam))

def EstLam2(hurdles, breaks):
    '''
    function to return log likelihood function instance
    inputs:
        hurdles: output of HurdleFun
        breaks: list of breakpoints
    output: function instance for calcluating log_likelihood given the setup(hurdles, breaks)
    
    TODO: Convert into decorator function
    '''
    
    total_users = sum(hurdles)
    # keep track of m0, m1 and m2
    m0 = hurdles[0]
    bp1 = breaks[0]
    m2 = hurdles[-1]
    bp_last = breaks[-1]
    m1 = total_users - m0 - m2
    
    def log_likehood(lam):
        """
        Specialized function to be called as a lambda, which takes the lam list and
        returns the log_likelihood
        
        """
        log_like = (m0 * np.log(cdf(lam, bp1))) + (m2 * -1*lam*bp_last)
        # If there are users in m1, then add relevant sums to log likelihood
        if m1 != 0:
            for i in range(len(breaks) - 1):
                log_like += hurdles[i+1]*np.log(cdf(lam, breaks[i + 1]) - cdf(lam, breaks[i]))
        return log_like
    
    return log_likehood

def MaxMLE(survival_list, breakpoints, lambda_list):
    """
    Given the list of survival of users in the form of the output of hurdlefun, breakpoints list and 
    the possible values of lambda, outputs the best lambda for which the MLE estimates are lowest
    Does that by using the EstLam2 function to get the MLE estimate
    
    Input: Survival list of users [], breakpoints [], possible lambda values []
    Output: best lambda float
    """
    PRT = EstLam2(survival_list, breakpoints)
    mle_list = [PRT(x) for x in lambda_list]
    index = np.argmax(mle_list)
    
    return lambda_list[index]

# Q1(a)

# Simulate 1,000 users with lambda = 2
n = 1000
parameter = 2
user_pdf = UserSim(n, parameter)  # simulate the probability densities of users

# graph funnel for 1000 users
survivors, bin_edges = get_survivors(user_pdf)
fig, ax = plt.subplots(1,1)
plt.bar(bin_edges, survivors, width=0.2, align="center")
plt.xlabel('Time')
plt.ylabel('Number of Users Remaining')
plt.title('Funnel Simulation of 1,000 Users; lambda = 2')
plt.grid(axis='y')
fig.savefig('funnelq1.png', dpi=100)

# Q1(b)

# generate exp rv with the respect of each lambda value
params_all = np.arange(0.2, 3.2, 0.2)
user_pdfs_all = [get_survivors(UserSim(n, parameter)) for parameter in params_all]

# graph funnel for different number of users
fig, axes = plt.subplots(5,3, sharex=False, sharey=False)
fig.set_size_inches(13, 14)
for i, ax in enumerate(axes.flatten()):
    ax.bar(user_pdfs_all[i][1], user_pdfs_all[i][0], width=0.2, align="center")
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Users Remaining')
    ax.grid(axis='y')
    ax.set_title('Funnel Simulation with Lambda = {:.2f}'.format(params_all[i]))
    
plt.tight_layout()
fig.savefig('funnelq2.png', dpi=100)


#Q2(b)

user_quitting_time = UserSim(1000, 1)

lambda_ = EstLam1(user_quitting_time)
print "Estimated lambda is {:.3f}".format(lambda_)

#Q2(c)

n = 500
lower, upper = bootstrap(n,quitting_time=user_quitting_time, alpha=0.5)
print('95% confidence interval for the estimated lambda is: [{:.3f},{:.3f}]'.format(lower,upper))


num_users = [100, 200, 500, 1000, 2000, 5000, 10000]
estimated_lambdas = []
lower_bound = []
upper_bound = []

n = 500
for each in num_users:
    user_quitting_time = UserSim(each, 1) # simulate user quitting time
    estimated_lambdas.append(EstLam1(user_quitting_time)) # calculate estimated lambda 
    
    lower, upper = bootstrap(n,quitting_time=user_quitting_time, alpha=0.5) # lower and upper bound of estimated lambda 
    lower_bound.append(lower)
    upper_bound.append(upper)

table = pd.DataFrame({'number of users': num_users, 
                      'estimated lambda': estimated_lambdas,
                      'lower bound': lower_bound,
                      'upper bound': upper_bound})

table = table[['number of users','estimated lambda','lower bound','upper bound']]

print table


# graph confidence interval

lambda_value = table['estimated lambda']
number_users = table['number of users']
lower_bound = table['lower bound']
upper_bound = table['upper bound']

fig, ax = plt.subplots(1,1)
plt.plot(number_users, lambda_value, 'black')
plt.plot(number_users,lower_bound , 'red')
plt.plot(number_users, upper_bound, 'blue')
plt.xlim(10, np.max(np.array(number_users)))
plt.legend(['lambda', 'lower bound', 'upper bound'])
fig.savefig('funnelq3.png', dpi=100)


# Q4(a)

# Set seet for reproducability
np.random.seed(42)

final_df = pd.DataFrame(columns=['breaks', 'lambda1', 'lambda2', 'diff'])
# Calcualate the difference in the estimates of lambda by EstLam1 and EstLam2
for b, breaks in enumerate([[0.0001,0.3],[0.25,0.3],[.25,1],[.25,5],[.25,10],[2,10],[4,10],[5,10],[.25,5,50],[4,10,50]],1):
    
    lambda_diff = []
    lambda1=[]
    lambda2=[]
    
    for i in range(0,1000):
        
        samples = UserSim(100, 1)
        lmbda1 = EstLam1(samples)
        lmdba2 = MaxMLE(HurdleFun(samples, breaks), breaks, list(np.arange(.1, 3, .05)))
        lambda1.append(lmbda1)
        lambda2.append(lmdba2)
        diff=lmbda1-lmdba2
        lambda_diff.append(diff)
    final_df.loc[b]=[breaks, np.round(np.mean(lambda1),4), np.round(np.mean(lambda2),4), np.round(np.mean(lambda_diff),4)]

print final_df
