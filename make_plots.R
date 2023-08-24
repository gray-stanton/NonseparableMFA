library(tidyverse)
setwd("/home/gray/code/")


figwidth <- 3.5*1.4
figheight <- 3.5*1.4
convdf <- read_csv("obj_by_iteration.csv")
convdf$niter = 1:nrow(convdf)
convlong <- convdf %>% gather("run", "obj", x1:x15)
p1 <- convlong %>%  filter(niter<=21) %>% ggplot(aes(x=(niter-1), y=obj, group=run)) + geom_line(color="steelblue") +
  theme_bw() + labs(x="Iterations (s)", y="l_T Objective") + scale_x_continuous(limits = c(0, 20), breaks=c(0, 5, 10, 15, 20))
  

ggsave("/home/gray/code/NonsepMFAJulia/convplot.pdf", p1, pdf(width=figwidth, height=figheight))



nobschangedf <- read_csv("changing_nobs_NMSE2.csv")


colvals <- c("depnmse" = "coral3", indepnmse = "springgreen4", sampcovnmse="slateblue")
shapevals <- c("depnmse" = 1, indepnmse=2, sampcovnmse=3)

p2 <- nobschangedf %>% gather("series", "nmse", depnmse:sampcovnmse) %>%
  mutate(nmsedb = 10*log(nmse, 10)) %>% group_by(series, nobs) %>% 
  summarize(mnmse = mean(nmsedb)) %>%
  ggplot(aes(x=nobs, y=mnmse, color=series, shape=series)) + geom_point() + geom_line() +
  theme_bw() + scale_x_continuous(limits=c(100, 600), breaks=c(200, 400, 600)) +
  scale_y_continuous(limits=c(-20 ,-3), breaks=c(-5, -10, -15, -20)) +
  labs(x="Number of Observations T", y="NMSE (dB)", color="Estimator", shape="Estimator") + theme(legend.position = c(0.8, 0.8)) +
  scale_color_manual(values=colvals, labels=c("Dep. MFA", "Indep. MFA", "Sample Cov.")) + scale_shape_manual(values=shapevals,labels=c("Dep. MFA", "Indep. MFA", "Sample Cov."))+
  theme(legend.box.background = element_rect(color="grey40", size=1.0))

 


ggsave("/home/gray/code/NonsepMFAJulia/varynobs.pdf", p2, pdf(width=figwidth, height=figheight))




nobschangedf2 <- read_csv("changing_nobs_NMSE_indep2.csv")


colvals <- c("depnmse" = "coral3", indepnmse = "springgreen4", sampcovnmse="slateblue")

p25 <- nobschangedf2 %>% gather("series", "nmse", depnmse:sampcovnmse) %>%
  mutate(nmsedb = 10*log(nmse, 10)) %>% group_by(series, nobs) %>% 
  summarize(mnmse = mean(nmsedb)) %>%
  ggplot(aes(x=nobs, y=mnmse, color=series, shape=series)) + geom_point() + geom_line() +
  theme_bw() + scale_x_continuous(limits=c(100, 600), breaks=c(200, 400, 600)) +
  scale_y_continuous(limits=c(-21 ,-3), breaks=c(-5, -10, -15, -20)) +
  labs(x="Number of Observations T", y="NMSE (dB)", color="Estimator", shape="Estimator") + theme(legend.position = c(0.8, 0.8)) +
  scale_color_manual(values=colvals, labels=c("Dep. MFA", "Indep. MFA", "Sample Cov.")) + scale_shape_manual(values=shapevals,labels=c("Dep. MFA", "Indep. MFA", "Sample Cov.")) + theme(legend.box.background = element_rect(color="grey40", size=1.0))



ggsave("/home/gray/code/NonsepMFAJulia/varynobsindep.pdf", p25, pdf(width=figwidth, height=figheight))


archangedf <- read_csv("changing_AR_nmsesdf.csv")
p3 <- archangedf %>%  gather("series", "nmse", depnmse:sampcovnmse) %>%
  mutate(nmsedb = 10*log(nmse, 10)) %>% group_by(series, arval) %>% summarize(mnmse = mean(nmsedb)) %>%
  ggplot(aes(x=arval, y=mnmse, color=series, shape=series)) + geom_point() + geom_line() + theme_bw() + scale_x_continuous(limits=c(0.0, 1.0), breaks=c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  labs(x="AR(1) Coefficient", y="NMSE (dB)", color="Estimator", shape="Estimator") + theme(legend.position = c(0.2, 0.8)) +
  scale_color_manual(values=colvals, labels=c("Dep. MFA", "Indep. MFA", "Sample Cov.")) + scale_shape_manual(values=shapevals,labels=c("Dep. MFA", "Indep. MFA", "Sample Cov.")) +theme(legend.box.background = element_rect(color="grey40", size=1.0))
  


ggsave("/home/gray/code/NonsepMFAJulia/varyingar.pdf", p3, pdf(width=figwidth, height=figheight))


facerrordf<- read_csv("factor_errors.csv")
col = c()
p4 <- facerrordf %>%  gather("series", "nmse", fnmse:ifnmse) %>%
  mutate(nmsedb = 10*log(nmse, 10)) %>% group_by(series, nobs) %>% summarize(mnmse = mean(nmsedb, trim=0.2)) %>%
  ggplot(aes(x=nobs, y=mnmse, color=series, shape=series)) + geom_point() + geom_line() +
  theme_bw() + scale_x_continuous(limits=c(100, 600), breaks=c(200, 400, 600)) +
  scale_y_continuous(limits=c(-10,0), breaks=c(-5, -10, -15, -20))


ggsave("/home/gray/code/NonsepMFAJulia/depfacpred.pdf", p5, pdf(width=figwidth, height=figheight))


facpredsdf<- read_csv("factor_predictions.csv")
facpredsdf$t <- 1:nrow(facpredsdf)

col = c("rf1" = "black", "fp1" = "coral3", "ifp1" = "springgreen4")
lnt = c("rf1" = 1, "fp1" = 2, "ifp1" = 2)

p5 <- facpredsdf %>%  gather("series", "preds", rf1, fp1, ifp1) %>% filter(t<=200) %>% 
  filter(series %in% c("fp1", "rf1")) %>%
  ggplot(aes(x=t, y=preds, color=series, linetype=series))  + geom_line() +
  theme_bw() + scale_x_continuous(limits=c(0, 200), breaks=c(0,50, 100, 150, 200)) +
  scale_linetype_manual(values=lnt[c(1, 2)],  labels=c("True Factor", "Predicted Factor")) + 
  scale_color_manual(values=col[c(1, 2)], labels=c("True Factor", "Predicted Factor")) + labs(x="Time t", y= "Common factor", color="Series", linetype="Series") +
  theme(legend.position = c(0.2, 0.8)) +theme(legend.box.background = element_rect(color="grey40", size=1.0))


p6 <- facpredsdf %>%  gather("series", "preds", rf1, fp1, ifp1) %>% filter(t<=200) %>% 
  filter(series %in% c("ifp1", "rf1")) %>%
  ggplot(aes(x=t, y=preds, color=series, linetype=series))  + geom_line() +
  theme_bw() + scale_x_continuous(limits=c(0, 200), breaks=c(0,50, 100, 150, 200)) +
  scale_linetype_manual(values=lnt[c(1, 3)],  labels=c("True Factor", "Predicted Factor")) + 
  scale_color_manual(values=col[c(1, 3)], labels=c("True Factor", "Predicted Factor")) + labs(x="Time t", y= "Common factor", color=element_blank(), linetype=element_blank()) +
  theme(legend.position = c(0.2, 0.8)) +theme(legend.box.background = element_rect(color="grey40", size=1.0), legend.title=element_blank())



ggsave("/home/gray/code/NonsepMFAJulia/depfacpred.pdf", p5, pdf(width=3.5*1.4, height=figheight))
ggsave("/home/gray/code/NonsepMFAJulia/indepfacpred.pdf", p6, pdf(width=3.5*1.4, height=figheight))




spderrs <- read_csv("temporal_nmse.csv")
p7 <- spderrs %>% group_by(nobs) %>% summarize(ml2 = mean(l2err), mli = mean(linferr),mbest=mean(l2best)) %>%
  ggplot(aes(x=nobs, y=ml2)) + geom_point(color="coral3") + geom_line(color="coral3") +
  theme_bw() + labs(x="Number of Observations T", y= "L2 Error") + lims(y=c(0.0, 0.5)) +
  geom_hline(aes(yintercept=mbest), color="grey40", linetype="dashed")


p8 <- spderrs %>% group_by(nobs) %>% summarize(ml2 = mean(l2err), mli = mean(linferr), mbest=mean(linfbest)) %>%
  ggplot(aes(x=nobs, y=mli)) + geom_point(color="coral3") + geom_line(color="coral3") +
  theme_bw() + labs(x="Number of Observations T", y= "Max Norm Error") + lims(y=c(0.0, 1.5)) + 
   geom_hline(aes(yintercept=mbest), color="grey40", linetype="dashed")

ggsave("/home/gray/code/NonsepMFAJulia/l2temperr.pdf", p7, pdf(width=figwidth, height=figheight))
ggsave("/home/gray/code/NonsepMFAJulia/linftemperr.pdf", p8, pdf(width=figwidth, height=figheight))



covfits <- read_csv("covfits.csv")
covfits$h <- 1:nrow(covfits) - 1
p9 <- covfits %>% filter(h<=15) %>% ggplot(aes(x=h)) + geom_point(aes(y=truecov1), color="grey40") + geom_line(aes(y=truecov1), color="grey40", linetype="dashed") +
  geom_point(aes(y=ftcov1), color="coral3") + geom_line(aes(y=ftcov1), color="coral3") + theme_bw() + labs(x="Lag h", y="Autocovariance")
p10 <- covfits %>% filter(h<=15) %>% ggplot(aes(x=h)) + geom_point(aes(y=truecov2), color="grey40") + geom_line(aes(y=truecov2), color="grey40", linetype="dashed") +
  geom_point(aes(y=ftcov2), color="coral3") + geom_line(aes(y=ftcov2), color="coral3") + theme_bw() + labs(x="Lag h", y="Autocovariance")


ggsave("/home/gray/code/NonsepMFAJulia/covfit1.pdf", p9, pdf(width=figwidth, height=figheight))
ggsave("/home/gray/code/NonsepMFAJulia/covfit2.pdf", p10, pdf(width=figwidth, height=figheight))
