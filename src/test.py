from IPython.parallel import Client

rc = Client(profile='local_cluster')
dview = rc[:]

with dview.sync_imports():
    import sys

def parallel(x):
    #import sys
    return '/Users/alex/mnist/src' in sys.path

print 'Local: ', parallel(1)
print 'Remote: ', str(dview.map_sync(parallel, range(1)))
