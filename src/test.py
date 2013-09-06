from IPython.parallel import Client
import sys

try:
    rc = Client( profile=sys.argv[1] )
    dview = rc[:]
except:
    rc = Client()
    dview = rc[:]

with dview.sync_imports():
    import sys
    import autoencoder

def parallel(x):
    #import sys
    da = autoencoder.dA(10,10)
    return '/Users/alex/mnist/src' in sys.path or '/home/susemihl/mnist/src' in sys.path

print 'Local: ', str(map(parallel,range(1)))
print 'Remote: ', str(dview.map_sync(parallel, range(1)))
