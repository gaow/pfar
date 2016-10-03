


###################    Bug Fix  PFA  ##################################

node1 <- c(2, 2, 5, 100, 40, 1, 25, 5, 10, 0);
sd <- c(2, 2, 10, 10, 4, 0.6, 30, 20, 0.05, 50)
node2 <- c(15, 5, 10, 50, 30, 20, 4, 0.4, -2, 10);

data <- as.numeric()

seq_val <- seq(0,1,length.out = 100)

for(q in 1:length(seq_val)){
  temp <- array(0,length(node1));
  for(n in 1:length(node1)){
    temp[n] <- rnorm(1,(1-seq_val[q])*node1[n] + (seq_val[q])*node2[n],sd[n]);
  }
  data <- rbind(data, temp)}

Fstart1 <- runif(10, 0, 500) 
Fstart2 <- runif(10, 0, 500) 
Fstart <- rbind(Fstart1, Fstart2)
control <- list(maxiter=2, logfile = 'file.txt')
out <- pfa(data, K=2, F=Fstart, control = control)

out$F



