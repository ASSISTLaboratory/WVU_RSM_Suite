# s=lhsu(xmin,xmax,nsample)
# LHS from uniform distribution
# Input:
#   xmin    : min of data (1,nvar)
#   xmax    : max of data (1,nvar)
#   nsample : no. of samples
# Output:
#   s       : random sample (nsample,nvar)
#   Budiman (2003)

# libraries
import numpy

def lhsu(xmin,xmax,nsample):
    """
    This function ....

    """
    nvar = xmin.shape[0]
    ran = numpy.random.rand(nsample,nvar)
    s = numpy.zeros((nsample,nvar))

    idx = numpy.arange(1,nsample+1)

    for j in range(nvar):
        numpy.random.shuffle(idx)
        P =(idx-ran[:,j])/numpy.float(nsample)
        s[:,j] = xmin[j] + P*(xmax[j]-xmin[j])
    # end

    return s

# end
