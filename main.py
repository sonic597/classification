import math
import matplotlib.pyplot as plt
import csv
import random

with open("DATA.csv") as dt: #imports the data, 1st row as headers are excluded
    c = list(csv.reader(dt))[1:]
    test_x=[float(jl[1]) for jl in c]
    test_y=[float(jl[0]) for jl in c]

# Comment out next 2 lines to use CSV data rather than test data.
test_x = [1,2,3,4,5,6,7,8,9,10] 
test_y = [0,0,0,1,1,1,1,0,0,0]

#shuffle data
_t = list(zip(test_x,test_y))
random.shuffle(_t)
test_x, test_y = zip(*_t)
test_x = list(test_x)
test_y = list(test_y)

#div. by range to normalise x values
ran = max(test_x)-min(test_x)
test_x = [i/ran for i in test_x]

size = len(test_x) #size of data

#ADJUSTABLE PARAMETERS
order = 5 #order of hypothesis polynomial
batch = size # the number of datapoints to fit to under each iteration
lrate = 5 # learning rate (by how much each set of feedback affects the hypothesis)
iters = 1000 # the number of iterations
detail = 50 # number of sample points in the hypothesis plot

thetas= [1]*(order+1) #initialising the vector for coefficients in the hypothesis polynomial

def sigmoid(x): # normalisation function to clamp values between 0 and 1
    return 1/(1+math.exp(-x))

def hyp(x): #hypothesis function
    h = 0
    for i in range(order+1):
        h+= x**i * thetas[i]
    return sigmoid(h)


ran = max(test_x)-min(test_x) #formatting x values to a smaller range 
inc = ran/detail
scdata = [min(test_x)+inc*i for i in range(detail+1)]

costs=[] #cost function outputs
for iterations in range(iters):
    for i in range(0,size, batch):
        xs = test_x[i: i+batch]
        ys = test_y[i: i+batch]
        ts = len(xs) #test size
        updates = [0]*(order+1)
        cost = 0 #COST (only for analysis)
        for datapoint in range(0,ts):
            _x = xs[datapoint]
            _y = ys[datapoint]
            _hypo = hyp(_x)
            _f = _hypo-_y
            for j in range(order+1): #update each theta
                th = _f*(_x**j) #because each "Feature" is just x to increasing powers
                updates[j] += th
        
            if _y == 1:                       #costs for analysis
                cost -= math.log(_hypo)       #costs for analysis
            elif _y == 0:                     #costs for analysis
                    cost -= math.log(1-_hypo) #costs for analysis  
        _g= (lrate/ts)
        for tindex in range(order+1): 
            thetas[tindex] -= _g*updates[tindex]
        
        costs.append(cost/ts) #costs for analysis

#PLOTTING 
plt.subplot(2,1,1)
plt.plot(test_x, test_y, 'o')
plt.plot(scdata, [hyp(i) for i in scdata])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Hypothesis and data")
plt.subplot(2,1,2)
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost over iterations")
plt.tight_layout()
plt.show()
