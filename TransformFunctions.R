##
## Function to transform to independence
##

transform_to_ind <- function(formula,
                             trainData,
                             trainLocs,
                             testData, #Don't include response
                             testLocs,
                             MaternParams=NULL, #Either null or a 2 vector of (rng, nug)
                             smoothness=1/2,
                             M = 30, #num neighbors
                             ncores=detectCores()-10){
  
  ######################################
  ## Figure out the nearest neighbors ##
  ######################################
  nnList <- mkNNIndx(trainLocs, m=M)
  
  #################################################
  ## Estimate a range and nugget if not provided ##
  #################################################
  if(is.null(MaternParams)){
    mFit <- fit.NN.Matern(formula, data=trainData, locs=trainLocs, nu=smoothness,
                          NearNeighs=nnList, num.cores=ncores)
    range <- 1/mFit$decay
    nugget <- mFit$nugget
  } else {
    range <- MaternParams[1]
    nugget <- MaternParams[2]
  }
  
  
  ################################
  ## Transform the Training Set ##
  ################################
  
  ## Define X and y matrices
  Xtrain <- model.matrix(formula, data=trainData)
  ytrain <- as.matrix(trainData[,all.vars(formula)[1]], ncol=1)
  
  ## Apply decorrelating transform by location
  indData <- mclapply(1:nrow(Xtrain), FUN=function(idx){
    if(idx==1){
      y <- ytrain[idx]
      w <- 1
      X <- Xtrain[idx,] / sqrt(w)
    } else if(idx==2){
      D <- rdist(trainLocs[1:idx,])
      R <- (1-nugget)*Matern(D, nu=1/2, range=range) + 
        nugget*diag(nrow(D))
      w <- as.numeric(1-R[1,-1]%*%solve(R[-1,-1])%*%R[-1,1])
      X <- (t(Xtrain[idx,]) - (R[1,-1]%*%solve(R[-1,-1])%*%Xtrain[nnList[[idx]],])) / sqrt(w)
      y <- (ytrain[idx]-R[1,-1]%*%solve(R[-1,-1])%*%(ytrain[nnList[[idx]]]))/sqrt(w)
    } else {
      D <- rdist(trainLocs[c(idx,nnList[[idx]]),])
      R <- (1-nugget)*Matern(D, nu=1/2, range=range) + 
        nugget*diag(nrow(D))
      w <- as.numeric(1-R[1,-1]%*%solve(R[-1,-1])%*%R[-1,1])
      X <- (t(Xtrain[idx,]) - (R[1,-1]%*%solve(R[-1,-1])%*%Xtrain[nnList[[idx]],])) / sqrt(w)
      y <- (ytrain[idx]-R[1,-1]%*%solve(R[-1,-1])%*%(ytrain[nnList[[idx]]]))/sqrt(w)
    }
    
    return(list(y=y, X=X, w=w))
  }, mc.cores=ncores) # End mclapply()
  
  ## Apply decorrelating transform to test data
  Xtest <- model.matrix(~ ., data=testData)
  indTestData <- mclapply(1:nrow(Xtest), FUN=function(idx){
    D <- rdist(matrix(testLocs[idx,], nrow=1), trainLocs)
    theNeighbors <- order(D)[1:M]
    R <- rdist(rbind(testLocs[idx,],trainLocs[theNeighbors,]))
    R <- nugget*diag(M+1)+(1-nugget)*Matern(R, range=range, smoothness=smoothness)
    R12 <- R[1,-1]%*%chol2inv(chol(R[-1,-1]))
    w <- as.numeric(1-R12%*%R[-1,1])
    X <- (t(Xtest[idx,])-R12%*%Xtrain[theNeighbors,])/sqrt(w)
    return(list(backTrans=R12%*%matrix(ytrain[theNeighbors,], ncol=1), X=X, 
                w=w))
  }, mc.cores=ncores)
  
  ## Return transformed data
  outList <- list(trainData=data.frame(y=do.call(rbind, lapply(indData, function(x){x$y})),
                                       do.call(rbind, lapply(indData, function(x){x$X}))),
                  testData=data.frame(do.call(rbind, lapply(indTestData, function(x){x$X}))),
                  range=range,
                  nugget=nugget,
                  M=M,
                  formula=formula,
                  backTransformInfo=lapply(indTestData,function(x){x$X<-NULL
                  return(x)}))
  return(outList)
  
  
} # End spatial_to_ind function

back_transform_to_spatial <- function(preds, transformObj){
  
  spatialPreds <- preds*sapply(transformObj$backTransformInfo, function(x){x$w})+
    sapply(transformObj$backTransformInfo, function(x){x$backTrans})
  return(spatialPreds)
  
}

# load("../Linear Simulated Data/LinSimDataSet17.RData")
# transformTest <- transform_to_ind(formula=y~.,
#                                   trainData=trainData,
#                                   trainLocs=trainLocs,
#                                   testData=testData[,-1],
#                                   testLocs=testLocs)
#  
# back_transform_to_spatial(rnorm(nrow(testData)), transformTest)














##
## Functions to estimate nugget & range
##

# fxDir <- getSrcDirectory(function(x) {x})
# source(paste0(fxDir,"/mkNNIndx.R"))
# source(paste0(fxDir,"/fitMaternGP.R"))
library(spNNGP)

## coords: n x 2 matrix of locations
## m:      Number of Neighbors
## n.omp.threads: How many cores to use

mkNNIndx <- function(coords, m, n.omp.threads=1){
  
  get.n.indx <- function(i, m){
    i <- i-1
    if(i == 0){
      return(NA)
    }else if(i < m){
      n.indx.i <- i/2*(i-1)
      m.i <- i
      return((n.indx.i+1):((n.indx.i+1)+i-1))
    }else{
      n.indx.i <- m/2*(m-1)+(i-m)*m
      m.i <- m
      return((n.indx.i+1):((n.indx.i+1)+m-1))
    }
  }
  
  n <- nrow(coords)
  nIndx <- (1+m)/2*m+(n-m-1)*m
  nnIndx <- rep(0, nIndx)
  nnDist <- rep(0, nIndx)
  nnIndxLU <- matrix(0, n, 2)
  
  n <- as.integer(n)
  m <- as.integer(m)
  coords <- as.double(coords)
  nnIndx <- as.integer(nnIndx)
  nnDist <- as.double(nnDist)
  nnIndxLU <- as.integer(nnIndxLU)
  n.omp.threads <- as.integer(n.omp.threads)
  
  out <- .Call("mkNNIndx", n, m, coords, nnIndx, nnDist, nnIndxLU, n.omp.threads)
  
  n.indx <- as.integer(nnIndx)
  
  n.indx.list <- vector("list", n)
  n.indx.list[1] <- NA
  for(i in 2:n){
    n.indx.list[[i]] <- n.indx[get.n.indx(i, m)]+1
  }
  n.indx.list
}




library(LatticeKrig)
library(parallel)
library(magrittr)

## Fit Spatial NN Model using ML
fit.NN.Matern <- function(formula,locs,nu,gridsize=15,NearNeighs,
                          num.cores=detectCores(),data=NULL){
  
  ## Assign variables
  X <- model.matrix(formula,data=data)
  y <- matrix(model.frame(formula,data=data)[,1],ncol=1)
  n <- nrow(X)
  if(length(gridsize)==1){
    sr.gridsize <- gridsize
    pct.gridsize <- gridsize
  } else {
    sr.gridsize <- gridsize[1]
    pct.gridsize <- gridsize[2]
  }
  
  ## Order the locations
  if(is.null(dim(locs))){
    locs <- matrix(locs, ncol=1)
  }
  ord <- GPvecchia::order_maxmin_exact(locs) 
  locs <- locs[ord,]
  y <- matrix(y[ord], ncol=1)
  X <- matrix(X[ord,], ncol=ncol(X))
  
  ## Create a Sequence for Spatial Range
  D <- rdist(locs[sample(n, size=min(n,500)),])
  max.dist <- max(D)
  min.dist <- max(apply(D,1,function(x){sort(x)[2]}))
  upperbound.decay <- 1/Matern.cor.to.range(min.dist,nu=nu,cor.target=0.05)
  lowerbound.decay <- 1/Matern.cor.to.range(max.dist,nu=nu,cor.target=0.95)
  #c(lowerbound.decay,upperbound.decay)
  sr.seq <- seq(lowerbound.decay,upperbound.decay,length=sr.gridsize)
  
  ## Create a Sequence for %Spatial
  pct.spatial <- seq(0,.99,length=pct.gridsize)
  
  ## Expand pct and spatial range grid
  pct.sr.grid <- expand.grid(pct.spatial,sr.seq)
  
  ## Parse it out into a list for parallel processing
  aMw.list <- vector('list',nrow(pct.sr.grid))
  for(i in 1:length(aMw.list)){
    aMw.list[[i]] <- list(alpha=pct.sr.grid[i,2],omega=1-pct.sr.grid[i,1])
  }
  
  ## Function for calculating likelihoods that can be run in parallel
  getLL <- function(x){
    ## Transform to ind y & x
    getYX <- function(ind){
      if(ind==1){
        return(list(y_IID=y[ind],X_IID=X[ind,]))
      } else {
        R <- x$omega*diag(1+min(ind-1, length(NearNeighs[[ind]]))) +
          (1-x$omega)*Matern(rdist(locs[c(ind, NearNeighs[[ind]]),]), alpha=x$alpha)
        RiRn_inv <- R[1,-1]%*%chol2inv(chol(R[-1,-1]))
        Xtrans <- X[ind,]-RiRn_inv%*%X[NearNeighs[[ind]],]
        ytrans <- y[ind]-RiRn_inv%*%y[NearNeighs[[ind]]]
        return(list(y_IID=ytrans,X_IID=Xtrans))
      }
    }
    yx <- lapply(1:n, getYX)
    iidY <- matrix(sapply(yx, function(v){v$y_IID}), ncol=1)
    iidX <- lapply(yx, function(v){v$X_IID}) %>% do.call(rbind,.)
    
    ## Find bhat
    bhat <- solve(t(iidX)%*%iidX)%*%t(iidX)%*%iidY
    
    ## Find sig2hat
    sig2hat <- sum((iidY-iidX%*%bhat)^2)/n
    
    ## Get ll
    return(list(ll=sum(dnorm(iidY, iidX%*%bhat, sqrt(sig2hat),log=TRUE)),
                bhat=bhat,
                bse=diag(solve(t(iidX)%*%iidX)),
                sigma2=sig2hat,
                nugget=x$omega,
                decay=x$alpha
    ))
  }
  
  ## Apply likelihood function to each combo
  ll.list <- mclapply(aMw.list, getLL, mc.cores=num.cores)
  
  ## Find max(ll)
  all.ll <- sapply(ll.list,function(x){return(x$ll)})
  max.ll <- which.max(all.ll)
  ll.list <- ll.list[[max.ll]]
  coef.table <- data.frame(Estimate=ll.list$bhat,StdErr=sqrt(ll.list$bse*ll.list$sigma2),
                           TestStat=ll.list$bhat/sqrt(ll.list$bse*ll.list$sigma2),
                           PVal2Sided=2*pnorm(abs(ll.list$bhat/sqrt(ll.list$bse*ll.list$sigma2)),lower=FALSE))
  rownames(coef.table) = colnames(X)
  
  ## Return Info
  return(list(coefTable=coef.table,sigma2=ll.list$sigma2,nugget=ll.list$nugget,
              decay=ll.list$decay,loglike=ll.list$ll,
              response=y,locs=locs,nu=nu,X=X,frm=formula,
              n.neighbors=length(NearNeighs[[n]])))
}

predict.NN.Matern <- function(NNMaternModel,predlocs,newdata=NULL){
  
  ## Errors
  if(is.null(newdata) & length(NNMaternModel$coefTable$Estimate)>1){
    stop(paste("MaternModel indicates the use of covariates.",
               "Please supply covariates at prediction locations via newdata"))
  }
  
  ## Determine prediction X matrix
  if(is.null(newdata)){
    predModelMatrix <- model.matrix(predlocs~1)
  } else {
    predModelMatrix <- model.matrix(NNMaternModel$frm,data=newdata)
  }
  
  ## Get prediction point by point
  getPred <- function(ind){
    nn <- order(rdist(matrix(predlocs[ind,],ncol=ncol(NNMaternModel$locs)),
                      NNMaternModel$locs))[1:NNMaternModel$n.neighbors]
    nnLocs <- rbind(matrix(predlocs[ind,], ncol=ncol(NNMaternModel$locs)),
                    matrix(NNMaternModel$locs[nn,], ncol=ncol(NNMaternModel$locs)))
    nnR <- NNMaternModel$nugget*diag(nrow(nnLocs))+
      (1-NNMaternModel$nugget)*Matern(rdist(nnLocs), nu=NNMaternModel$nu, alpha=NNMaternModel$decay)
    pred <- predModelMatrix[ind,]%*%NNMaternModel$coefTable$Estimate +
      nnR[1,-1]%*%solve(nnR[-1,-1])%*%(NNMaternModel$response[nn]-
                                         matrix(NNMaternModel$X[nn,],ncol=ncol(NNMaternModel$X))%*%
                                         NNMaternModel$coefTable$Estimate)
    se.pred <- NNMaternModel$sigma2*(1-nnR[1,-1]%*%solve(nnR[-1,-1])%*%nnR[-1,1])
    return(list(pred=pred, se.pred=se.pred))
  }
  allPreds <- lapply(1:nrow(predlocs), getPred)
  
  return(data.frame(predlocs=predlocs,pred=sapply(allPreds, function(v){v$pred}),
                    se=sapply(allPreds, function(v){v$se.pred})))
  
}




