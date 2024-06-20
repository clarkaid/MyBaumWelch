#BaumWelchAlgo.R
#Author: Aidan Clark
#A simple implementation of the Baum-Welch Algorithm for (normal) Hidden Markov Models
#Date: 6/20/2024

#Note:
#Theta for a Hidden Markov Model (HMM), assuming normally distributed emissions,
#has four key parameters:
#I = initial probabilities for each state
#A = transition probabilities for each state
#mu = mean emission for each state
#sigma = variance of emissions for each state

#We work in log-space for much of our probability computations, for numerical stability


#Useful Functions

#log sum exp of two values
lse2 <- local({
  LOGEPS <- log(.Machine$double.eps / 2) # cache
  function (x, y) {
    m <- pmax(x, y); d <- -abs(x - y)
    ifelse(d < LOGEPS, m, m + log(1 + exp(d))) # m + log1pe(d)
  }
})

#soft max (log-sum-exp) of vector `x`(for numerical stability)
lse <- function (x) Reduce(lse2, x)

#Takes a emmission value, a mean mu, and a standard deviation sigma and returns
#log of the prob of that emission value assuming a normal dist
log_emiss <- function(emission, mu, sigma){
  dnorm(x = emission, mean = mu, sd = sigma, log = TRUE)
}

#########################################################################

#Baum-Welch Implementation:

#Takes our current guess for theta and performs one iteration of the Baum-Welch algo
#Assume that cur_theta is a list of: logI, logA, mu, var, where:
#logI is a vector of length |S|
#logA is a table of size |S| x |S|
#mu and var are vectors of length |S|
# and S is the (finite) state space of the HMM
update_theta <- function(cur_theta, data){
  
  log_I = cur_theta$log_I
  log_A = cur_theta$log_A
  mu = cur_theta$mu
  sigma = cur_theta$sigma
  
  n = length(data) #Number of observations
  ns = length(log_I) #Number of states
  
  
  #Forward Pass
  log_alpha = matrix(nrow = n, ncol = ns) #Alpha table, all in logs
  
  #Base case: Set t = 1
  log_alpha[1, ] <- log_I + log_emiss(data[1], mu, sigma) #initial(i) * E[i, Y1]
  
  #Fill in rest of table, for each time t and each state j
  for (t in 2:n){
    for (s in 1:ns){
      #logE[i, Y_t] + lse over all s of (alpha_s(t-1) + A_is)
      log_alpha[t,s] = log_emiss(data[t], mu[s], sigma[s]) + lse(log_alpha[t-1,] + log_A[,s])
    }
  }
  
  #Backward Pass
  log_beta = matrix(nrow = n, ncol = ns)
  colnames(log_beta) <- rownames(cur_theta$logE)
  
  #Base Case: t = n
  log_beta[n,] = 0 #log(1) = 0
  
  #Fill in rest of table
  for (t in (n-1):1){
    for (s in 1:ns){
      log_beta[t,s] = lse(log_beta[t+1, ] + log_A[s,] + log_emiss(data[t+1], mu, sigma))
    }
  }
  
  #r and xi tables
  log_r = matrix(nrow = n, ncol = ns)
  #Fill in r table, where r[t,i] = P(X_t = i | Y, theta)
  for (t in 1:n){
    for (s in 1:ns){
      log_r[t,s] = log_alpha[t,s] + log_beta[t,s] - lse(log_alpha[t,] + log_beta[t,])
    }
  }
  
  #3d table of size nsxnsx(n-1)-- n-1 because we can't go from T to T+1
  log_xi = array(dim = c(ns, ns, n-1))
  for (t in 1:(n-1)){
    
    #Calculate normalizing constant for this t
    counter = 1
    lis = numeric(9)
    for (k in 1:ns){
      for (w in 1:ns){
        lis[counter] = log_alpha[t,k] + log_A[k,w] + log_beta[t+1, w] + log_emiss(data[t+1], mu[w], sigma[w])
        counter = counter + 1
      }
    }
    normalizer = lse(lis)
    
    for (i in 1:ns){
      for (j in 1:ns){
        log_xi[i,j,t] = log_alpha[t,i] + log_A[i,j] + log_beta[t+1, j] + log_emiss(data[t+1], mu[j], sigma[j]) - normalizer
      }
    }
  }
  
  #Updates
  new_log_I = log_r[1,] #B/c r_i(1) = P(X_1 = i | Y, theta)
  
  new_log_A = matrix(nrow = ns, ncol = ns)
  for (i in 1:ns){
    for (j in 1:ns){
      new_log_A[i,j] = lse(log_xi[i,j,]) - lse(log_r[,i])
    }
  }
  
  new_mu = numeric(ns)
  for (s in 1:ns){
    new_mu[s] = sum(exp(log_r[,s]) * data) / exp(lse(log_r[,s]))
  }
  
  new_sigma = numeric(ns)
  for (s in 1:ns){
    new_sigma[s] = sqrt( sum(exp(log_r[,s]) * (data - mu[s])^2)      / exp(lse(log_r[,s])))
  }
  
  new_theta = list(
    log_I = new_log_I,
    log_A = new_log_A,
    mu = new_mu,
    sigma = new_sigma
  )
  
}

#Wrapper function:
#Takes our data (emissions), an initial guess for theta, and a max number of iterations
#and performs that many iterations of the Baum-Welch algorithm
#Returns estimates of theta
#Each value of theta is stored as a list of logI, logA, mu, and var
#Data is emissions
baum_welch <- function(data, initial_theta, max_iter = 100){
  
  cur_theta = initial_theta
  for (iter in 1:max_iter){
    cur_theta = update_theta(cur_theta, data)
  }
  
  return (cur_theta)
  
}