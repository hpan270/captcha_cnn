acc.df=read.csv('YeModel100k200e3l.csv', header = T)
acc.df2=read.csv('YeModel100k200e5L.csv', header = T)
col=c('cyan4','coral3')
#opar=par(mfrow=c(2,2))
plot(acc.df$val_loss,type='l', col=col[1], main='CNN Train Loss (3Layers)',ylab='Crossentropy Loss', las=1
     ,xlab='Epoch', lwd=2, lty=1)
lines(acc.df$loss,col=col[2],lwd=2, lty=2)
legend('topright',lty=1:2,col=col,lwd=2,legend=c('Val_Loss','Train_Loss'))

plot(acc.df$val_acc,type='l', col=col[1], main='CNN Train Accuracy (3Layers)',ylab='Accurate Rate', las=1
     ,xlab='Epoch', lwd=2, lty=1)
lines(acc.df$acc,col=col[2],lwd=2, lty=2)
legend('topright',lty=1:2,col=col,lwd=2,legend=c('Val_Acc','Train_Acc'))

plot(acc.df2$val_loss,type='l', col=col[1], main='CNN Train Loss (5Layers)',ylab='Crossentropy Loss', las=1
     ,xlab='Epoch', lwd=2, lty=1)
lines(acc.df2$loss,col=col[2],lwd=2, lty=2)
legend('topright',lty=1:2,col=col,lwd=2,legend=c('Val_Loss','Train_Loss'))

plot(acc.df2$val_acc,type='l', col=col[1], main='CNN Train Accuracy (5Layers)',ylab='Accurate Rate', las=1
     ,xlab='Epoch', lwd=2, lty=1)
lines(acc.df2$acc,col=col[2],lwd=2, lty=2)
legend('topright',lty=1:2,col=col,lwd=2,legend=c('Val_Acc','Train_Acc'))

#####
plot(acc.df$val_loss[1:80],type='l', col=col[1], main='CNN Train Loss (80 Epochs)',ylab='Crossentropy Loss', las=1
     ,xlab='Epoch', lwd=2, lty=1)
lines(acc.df$loss[1:80],col=col[2],lwd=2, lty=2)
legend('topright',lty=1:2,col=col,lwd=2,legend=c('Val_Loss','Train_Loss'))

plot(acc.df$val_acc[1:80],type='l', col=col[1], main='CNN Train Accuracy (80 Epochs)',ylab='Accurate Rate', las=1
     ,xlab='Epoch', lwd=2, lty=1)
lines(acc.df$acc[1:80],col=col[2],lwd=2, lty=2)
legend('topright',lty=1:2,col=col,lwd=2,legend=c('Val_Acc','Train_Acc'))
#par(opar)
colMeans(acc.df[41:60,]);colMeans(acc.df2[41:60,])
sum(acc.df$sec)/60/60;sum(acc.df2$sec)/60/60
######
plot(x=1:200,y=acc.df[,6], type='l', col='cyan4',lwd=2, lty=1, main='Validation Accuracy', las=1,
     xlab='Epoch',ylab='Accurate Rate')
lines(x=1:200,y=acc.df2[,6], col='dodgerblue3',lwd=2, lty=2)
legend('topright',lty=1:2,col=c('cyan4','dodgerblue3'),lwd=2,
       legend=c('3 Conv2D Layers','5 Conv2D Layers'))

plot(x=1:200,y=acc.df[,5], type='l', col='cyan4',lwd=2, lty=1, main='Validation Loss', las=1,
     xlab='Epoch',ylab='Crossentropy Loss')
lines(x=1:200,y=acc.df2[,5], col='dodgerblue3',lwd=2, lty=2)
legend('topright',lty=1:2,col=c('cyan4','dodgerblue3'),lwd=2,
       legend=c('3 Conv2D Layers','5 Conv2D Layers'))

#########
cl.df=read.csv('results_CLbest.csv',header = F)
colnames(cl.df)=c('FileName','Pred','TimeMSec','TrueLabel','PredLabel')
head(cl.df)
mean(cl.df$TrueLabel==cl.df$PredLabel)  #41.5%, 2hours, 10 epochs//0.982, 10 hours, 64 epochs
m=table(cl.df$PredLabel,cl.df$TrueLabel)  
t=rbind(diag(m),c(0,0,0,0,0))
t[2,]=200-t[1,]
rownames(t)=c('Correct','Wrong')
colnames(t)=paste0('L',2:6)
tt=t/200
bp=barplot(tt[1,],col='dodgerblue3', ylim=c(0,1.1), las=1, 
           main='CNN Predict Length of Captcha\n(Overall Prediction Acc 98.2%)',
        ylab="Accuracy Rate", xlab='Length of Captcha (200 images each group)')
text(x=bp,labels=paste0(tt[1,]*100,'%'), y=0.9, col=c(rep('white',3),'white','white'),font=2)
box(which = "plot", lty = "solid")
#predict time
mean(cl.df[,3])

cltrain.df=read.csv('train_CL64E.csv', header = T)
col=c('cyan4','coral3')
plot(cltrain.df$val_loss,type='l', col=col[1], 
     main='CNN Train Loss\nMean(Val_Loss)=0.093 after 30 Epochs',ylab='Crossentropy Loss', las=1,
     sub='Predicting Length of CAPTCHA'
     ,xlab='Epoch', lwd=2, lty=1)
lines(cltrain.df$loss,col=col[2],lwd=2, lty=2)
legend('topright',lty=1:2,col=col,lwd=2,legend=c('Val_Loss','Train_Loss'))

plot(cltrain.df$val_acc,type='l', col=col[1], 
     main='CNN Train Accuracy\nMean(Val_Acc)=96.4% after 30 Epochs',ylab='Accurate Rate', las=1
     ,sub='Predicting Length of CAPTCHA'
     ,xlab='Epoch', lwd=2, lty=1)
lines(cltrain.df$acc,col=col[2],lwd=2, lty=2)
legend('bottomright',lty=1:2,col=col,lwd=2,legend=c('Val_Acc','Train_Acc'))

colMeans(cltrain.df[30:64,])
sum(cltrain.df$sec)/60/60
