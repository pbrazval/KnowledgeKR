import DirichletForest as DF
from numpy import *

# Model parameters (see paper for meanings)
(alpha,beta,eta) = (1, .01, 100)

# Number of topics, size of vocab
(T,W) = (2,3)

# Vocabulary
vocab = ['apple','banana','motorcycle']

# Read docs
docs = DF.readDocs('example.docs')

# Build DF, apply constraints 
df = DF.DirichletForest(alpha,beta,eta,T,W,vocab)

# Must-Link between apple and banana
df.merge('apple','banana')

# Cannot-Link between apple and motorcycle
df.split('apple','motorcycle')

# Do inference on docs
(numsamp, randseed) = (50, 821945)
df.inference(docs,numsamp,randseed)

# #Output results
# print 'Top 3 words from learned topics'
# df.printTopics(N=3)
