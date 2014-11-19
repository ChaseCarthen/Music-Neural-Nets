# Author: Nicolas Boulanger-Lewandowski
# University of Montreal (2012)
# RNN-RBM deep learning tutorial
# More information at http://deeplearning.net/tutorial/rnnrbm.html

import glob
import os
import sys
from multiprocessing import Process, Manager, Lock
import multiprocessing
import numpy
from transposer import *
from music21 import *
try:
    import pylab
except ImportError:
    print "pylab isn't available, if you use their fonctionality, it will crash"
    print "It can be installed with 'pip install -q Pillow'"

from midi.utils import midiread, midiwrite
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#from theano import sandbox

# MAKE SURE TO IMPORT CUDA IN PROCESSES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#import theano.sandbox.cuda
#from theano import ProfileMode
#profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())

#Don't use a python long as this don't work on 32 bits computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

def gradient_updates_momentum(cost, params, learning_rate, momentum,v_sample):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param + param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update - learning_rate*T.grad(cost, (param),consider_constant=[v_sample])))
    return updates

def compute_function(gpu,args,weightQueue,inVecQueue,costQueue,l):
    #build thenao functions
    #return 
    import theano.tensor as T
    import theano
    import multiprocessing
    print "SIZE::::::::::::::::::::::",weightQueue.qsize(),args[0]['lr']
    params = weightQueue.get()
    #print params 
    weightQueue.put(params)
    W = theano.shared(params[0])
    bv = theano.shared(params[1])
    bh =  theano.shared(params[2])
    Wuh =  theano.shared(params[3])
    Wuv = theano.shared(params[4])
    Wvu = theano.shared(params[5])
    Wuu = theano.shared(params[6])
    bu = theano.shared(params[7])
    stop = args[0]['running']
    lr = args[0]["lr"]
    #lr = .001
    momentum = .5
    decay = .9
    import theano.sandbox.cuda
    pastp = params
    params= W,bv,bh,Wuh,Wuv,Wvu,Wuu,bu
    print "BEFORE CRAZY COMPILE",stop
    from theano.tensor.shared_randomstreams import RandomStreams
    from theano import sandbox
    theano.sandbox.cuda.use(gpu)
    (v, v_sample, cost, monitor, params, updates_train, v_t,
    updates_generate) = build_rnnrbm(W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu)

    #gradient = T.grad(cost, params, consider_constant=[v_sample])
    #updates_train[gradient] = gradient
    #updates_train.update(pastp)
    #updates_train.update(((p, p - lr * g) for p, g in zip(params,gradient)))
    print "HELLO"+str(v[0][0]) 
    updates_train.update(gradient_updates_momentum(cost, params, lr, momentum,v_sample))
    train_function = theano.function([v], [monitor],updates=updates_train)
    generate_function = theano.function([], v_t,updates=updates_generate)

    test_function = theano.function([v],[monitor],updates=updates_train)
    print "AFTER CRAZY COMPILE"
    step_size = 100
    #print generate_function()
    #return
    while(stop):
       #print "LOOPING"
       # update shared variables
       # This is where we need to process some stuff
       stop = args[0]["running"]
       #if weightQueue.qsize() > 1:
       #   print "OVERWEIGHT"
       if inVecQueue.qsize() > 0:
         #l.acquire()
         #print "TRAINING"

         p2 = weightQueue.get()
         #if weightQueue.empty():
         #   weightQueue.put(p2)
         #l.release()
         W.set_value(p2[0])
         bv.set_value(p2[1])
         bh.set_value(p2[2])
         Wuh.set_value(p2[3])
         Wuv.set_value(p2[4])
         Wvu.set_value(p2[5])
         Wuu.set_value(p2[6])
         bu.set_value(p2[7])
         while not inVecQueue.empty():
            vec = inVecQueue.get()
            total = 0
            # Batch size = 100 
            for j in xrange(0, len(vec),step_size):
              #l.acquire()

              #print "GIVING DATA"
              #count = count + 1
              #inVecs[s%8].put(sequence[j:j+batch_size])
              tcost = train_function(vec[j:j+step_size])
              #print tcost
              total = total + tcost[0]
              costQueue.put(tcost)
              #print tcost 
              out_params = (W.get_value()),(bv.get_value()),(bh.get_value()),(Wuh.get_value()),(Wuv.get_value()),(Wvu.get_value()),(Wuu.get_value()),(bu.get_value()),total
              #if weightQueue.qsize() > 1:
              #   print "OUCH"
              #if weightQueue.qsize() > 0:
              #    for i in xrange(0,weightQueue.qsize()):
              #       weightQueue.get()
              #l.acquire()
              #if weightQueue.full():
              #   weightQueue.get()

              #l.release()
              #print "SAME: ", a
 
            #out_params = (W.get_value()+p2[0])/2.0,(bv.get_value()+p2[1])/2.0,(bh.get_value()+p2[2])/2.0,(Wuh.get_value()+p2[3])/2.0,(Wuv.get_value()+p2[4])/2.0,(Wvu.get_value()+p2[5])/2.0,(Wuu.get_value()+p2[6])/2.0,(bu.get_value()+p2[7])/2.0
            o = args[0]

            #out_params = (W.get_value()),(bv.get_value()),(bh.get_value()),(Wuh.get_value()),(Wuv.get_value()),(Wvu.get_value()),(Wuu.get_value()),(bu.get_value()),total
            #out_params = (W.get_value()*p2[8]+p2[0]*total)/(total+p2[8]),(bv.get_value()*p2[8]+p2[1]*total)/(total+p2[8]),(bh.get_value()*p2[8]+p2[2]*total)/(total+p2[8]),(Wuh.get_value()*p2[8]+p2[3]*total)/(p2[8]+total),(Wuv.get_value()*p2[8]+p2[4]*total)/(total+p2[8]),(Wvu.get_value()*p2[8]+p2[5]*total)/(total+p2[8]),(Wuu.get_value()*p2[8]+p2[6]*total)/(p2[8]+total),(bu.get_value()*p2[8]+p2[7]*total)/(total+p2[8]),total
            o['params'] = out_params
            #m.list()
            #print p2[0],vec
            args[0] = o
            #costQueue.put(tcost)
         #if not weightQueue.full():

         weightQueue.put(out_params)
         #l.release()

def build_rbm(v, W, bv, bh, k):
    '''Construct a k-step Gibbs chain starting at v for an RBM.

v : Theano vector or matrix
  If a matrix, multiple chains will be run in parallel (batch).
W : Theano matrix
  Weight matrix of the RBM.
bv : Theano vector
  Visible bias vector of the RBM.
bh : Theano vector
  Hidden bias vector of the RBM.
k : scalar or Theano scalar
  Length of the Gibbs chain.

Return a (v_sample, cost, monitor, updates) tuple:

v_sample : Theano vector or matrix with the same shape as `v`
  Corresponds to the generated sample(s).
cost : Theano scalar
  Expression whose gradient with respect to W, bv, bh is the CD-k approximation
  to the log-likelihood of `v` (training example) under the RBM.
  The cost is averaged in the batch case.
monitor: Theano scalar
  Pseudo log-likelihood (also averaged in the batch case).
updates: dictionary of Theano variable -> Theano variable
  The `updates` object returned by scan.'''

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        theano.printing.Print('this is a very important value')(v)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX)


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return numpy.zeros(shape, dtype=theano.config.floatX)


def build_rnnrbm(W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu) :
    '''Construct a symbolic RNN-RBM and initialize parameters.

n_visible : integer
  Number of visible units.
n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.

Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
          updates_generate) tuple:

v : Theano matrix
  Symbolic variable holding an input sequence (used during training)
v_sample : Theano matrix
  Symbolic variable holding the negative particles for CD log-likelihood
  gradient estimation (used during training)
cost : Theano scalar
  Expression whose gradient (considering v_sample constant) corresponds to the
  LL gradient of the RNN-RBM (used during training)
monitor : Theano scalar
  Frame-level pseudo-likelihood (useful for monitoring during training)
params : tuple of Theano shared variables
  The parameters of the model to be optimized during training.
updates_train : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  training function.
v_t : Theano matrix
  Symbolic variable holding a generated sequence (used during sampling)
updates_generate : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  generation function.'''
     
    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu
    print params
    v = T.matrix()  # a training sequence
    # n_hidden_recurrent -- THIS NEEDS TO BE FIXED
    u0 = T.zeros((100,))  # initial value for the RNN hidden
                                         # units

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence(v_t, u_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv)
        bh_t = bh + T.dot(u_tm1, Wuh)
        generate = v_t is None
        if generate:
            # THIS NEEDS TO BE FIXED
            v_t, _, _, updates = build_rbm(T.zeros((89,)), W, bv_t,
                                           bh_t, k=25)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    (u_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
        sequences=v, outputs_info=[u0, None, None], non_sequences=params)
    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                     k=15)
    updates_train.update(updates_rbm)

    # symbolic loop for sequence generation
    (v_t, u_t), updates_generate = theano.scan(
        lambda u_tm1, *_: recurrence(None, u_tm1),
        outputs_info=[None, u0], non_sequences=params, n_steps=1000)

    return (v, v_sample, cost, monitor, params,updates_train, v_t,
            updates_generate)



def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
sequences.'''

    def __init__(self, n_hidden=150, n_hidden_recurrent=100, lr=0.001,
                 r=(21, 110), dt=0.3):
        '''Constructs and compiles Theano functions for training and sequence
generation.

n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.
lr : float
  Learning rate
r : (integer, integer) tuple
  Specifies the pitch range of the piano-roll in MIDI note numbers, including
  r[0] but not r[1], such that r[1]-r[0] is the number of visible units of the
  RBM at a given time step. The default (21, 109) corresponds to the full range
  of piano (88 notes).
dt : float
  Sampling period when converting the MIDI files into piano-rolls, or
  equivalently the time difference between consecutive time steps.'''

        self.r = r
        self.dt = dt
        self.n_hidden=n_hidden
        self.n_hidden_recurrent=100
        self.lr=0.001


        #return 
        # Process 
        params = runner(r[1]-r[0],n_hidden,n_hidden_recurrent)
        self.params = params
        #print params
        self.manager = Manager()
        self.args = self.manager.list()
        self.args.append({})
        shared_args = self.args[0]
        shared_args["params"] = params
        shared_args["out_params"] = params
        shared_args["running"] = True
        shared_args['lr'] = self.lr
        self.args[0] = shared_args
        #self.weightQueue = multiprocessing.Queue(20)
        #self.weightQueue.put(params)
        self.inVecQueue = multiprocessing.Queue()
        self.inVecQueue2 = multiprocessing.Queue()
        self.inVecQueue3 = multiprocessing.Queue()
        self.inVecQueue4 = multiprocessing.Queue()
        self.inVecQueue5 = multiprocessing.Queue()
        self.inVecQueue6 = multiprocessing.Queue()
        self.inVecQueue7 = multiprocessing.Queue()
        self.inVecQueue8 = multiprocessing.Queue()
        self.weightQueue = multiprocessing.Queue()
        self.weightQueue2 = multiprocessing.Queue()
        self.weightQueue3 = multiprocessing.Queue()
        self.weightQueue4 = multiprocessing.Queue()
        self.weightQueue5 = multiprocessing.Queue()
        self.weightQueue6 = multiprocessing.Queue()
        self.weightQueue7 = multiprocessing.Queue()
        self.weightQueue8 = multiprocessing.Queue()
        self.costQueue = multiprocessing.Queue()
        self.weightOutQueue = multiprocessing.Queue()
        self.l1 = Lock()
        self.l2 = Lock()
        self.l3 = Lock()
        self.l4 = Lock()
        self.l5 = Lock()
        self.l6 = Lock()
        self.l7 = Lock()
        self.l8 = Lock()
        self.pro = Process(target=compute_function,args=('gpu0',self.args,self.weightQueue,self.inVecQueue,self.costQueue,self.l1))
        self.pro2 = Process(target=compute_function,args=('gpu1',self.args,self.weightQueue2,self.inVecQueue2,self.costQueue,self.l1))
        self.pro3 = Process(target=compute_function,args=('gpu2',self.args,self.weightQueue3,self.inVecQueue3,self.costQueue,self.l1))
        self.pro4 = Process(target=compute_function,args=('gpu3',self.args,self.weightQueue4,self.inVecQueue4,self.costQueue,self.l1))
        #self.pro.start()
        self.pro5 = Process(target=compute_function,args=('gpu4',self.args,self.weightQueue5,self.inVecQueue5,self.costQueue,self.l1))
        self.pro6 = Process(target=compute_function,args=('gpu5',self.args,self.weightQueue6,self.inVecQueue6,self.costQueue,self.l1))
        self.pro7 = Process(target=compute_function,args=('gpu6',self.args,self.weightQueue7,self.inVecQueue7,self.costQueue,self.l1))
        self.pro8 = Process(target=compute_function,args=('gpu7',self.args,self.weightQueue8,self.inVecQueue8,self.costQueue,self.l1))
        self.pro.start()
        self.pro2.start()
        self.pro3.start()
        self.pro4.start()
        #self.weightQueue.put(params)
        self.pro5.start()
        self.pro6.start()
        self.pro7.start()
        self.pro8.start()
        self.weightQueue.put(params)
        self.weightQueue2.put(params)
        self.weightQueue3.put(params)
        self.weightQueue4.put(params)
        self.weightQueue5.put(params)
        self.weightQueue6.put(params)
        self.weightQueue7.put(params)
        self.weightQueue8.put(params)
        self.weights = [self.weightQueue,self.weightQueue2,self.weightQueue3,self.weightQueue4,self.weightQueue5,self.weightQueue6,self.weightQueue7,self.weightQueue8]
        '''self.weightQueue.put(params)
        self.weightQueue.put(params)
        self.weightQueue.put(params)
        self.weightQueue.put(params)
        self.weightQueue.put(params)
        self.weightQueue.put(params)
        self.weightQueue.put(params)'''
        print r[1] - r[0]
        '''self.W = params[0]
        self.bv = params[1]
        self.bh = params[2]
        self.Wuh = params[3]
        self.Wuv = params[4]
        self.Wvu = params[5]
        self.Wuu = params[6]
        self.bu = params[7]
        '''
        pastp = params
        self.W = theano.shared(params[0])
        self.bv = theano.shared(params[1])
        self.bh =  theano.shared(params[2])
        self.Wuh =  theano.shared(params[3])
        self.Wuv = theano.shared(params[4])
        self.Wvu = theano.shared(params[5])
        self.Wuu = theano.shared(params[6])
        self.bu = theano.shared(params[7])#'''
        params = self.W,self.bv,self.bh,self.Wuh,self.Wuv,self.Wvu,self.Wuu,self.bu
        print params
        (v, v_sample, cost, monitor, per ,updates_train, v_t,
        updates_generate) = build_rnnrbm(self.W, self.bv, self.bh, self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu)
 
        #gradient = T.grad(cost, per, consider_constant=[v_sample])
        #updates_train.update(((p, p - lr * g) for p, g in zip(per,gradient)))
        #print "HELLO"+str(v[0][0])
        #self.train_function = theano.function([v], monitor,updates=updates_train)
        #print "HELLO"+str(v[0][0])
        self.generate_function = theano.function([], v_t,updates=updates_generate)
        return
    def train(self, files, batch_size=100, num_epochs=200):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
files converted to piano-rolls.

files : list of strings
  List of MIDI files that will be loaded as piano-rolls for training.
batch_size : integer
  Training sequences will be split into subsequences of at most this size
  before applying the SGD updates.
num_epochs : integer
  Number of epochs (pass over the training set) performed. The user can
  safely interrupt training with Ctrl+C at any time.'''

        assert len(files) > 0, 'Training set is empty!' \
                               ' (did you download the data files?)'
        dataset = [midiread(f, self.r,
                            self.dt).piano_roll.astype(theano.config.floatX)
                   for f in files]
        print "DONE DOWNLOADING"
        #self.pro.start()
        try:
            count = 0
            processed = 0
            inVecs = [self.inVecQueue,self.inVecQueue2,self.inVecQueue3,self.inVecQueue4,self.inVecQueue5,self.inVecQueue6,self.inVecQueue7,self.inVecQueue8]
            for epoch in xrange(num_epochs):
                numpy.random.shuffle(dataset)
                costs = []
                ''' self.l1.acquire()
                self.l2.acquire()
                self.l3.acquire()
                self.l4.acquire()
                self.l5.acquire()
                self.l6.acquire()
                self.l7.acquire()
                self.l8.acquire()'''
                #ds = chunks(dataset, len(dataset)/4)
                count = 0
                processed = 0
                for s, sequence in enumerate(dataset):
                       inVecs[s%8].put(sequence)
                       #count = count + len(sequence)/100
                       for j in xrange(0, len(sequence), batch_size):
                       #print "GIVING DATA"
                          count = count + 1
                       #inVecs[s%8].put(sequence[j:j+batch_size])
                '''self.l1.release()
                self.l2.release()
                self.l3.release()
                self.l4.release()
                self.l5.release()
                self.l6.release()
                self.l7.release()
                self.l8.release()'''
                while processed != count:
                   if self.costQueue.qsize()>0:
                      cost = self.costQueue.get()
                      costs.append(cost)
                      processed = processed + 1
                sums = []
                items = self.weightQueue.qsize()
                #while self.weightQueue.qsize() > 0:
                #   d = self.weightQueue.get()
                #   sums.append(d)
                adder = [None,None,None,None,None,None,None,None]
                for i in xrange(0,8):
                   q = self.weights[i].get()
                   for j in xrange(0,8):
                      if adder[j] == None:
                         adder[j] = q[j]
                      else:
                         adder[j] += q[j]
                for i in range(0,8):
                   adder[i] /= float(8)
                p = adder[0],adder[1],adder[2],adder[3],adder[4],adder[5],adder[6],adder[7]
                if epoch != num_epochs:
                   self.weightQueue.put(p)
                   self.weightQueue2.put(p)
                   self.weightQueue3.put(p)
                   self.weightQueue4.put(p)
                   self.weightQueue5.put(p)
                   self.weightQueue6.put(p)
                   self.weightQueue7.put(p)
                   self.weightQueue8.put(p)   
                print 'Epoch %i/%i' % (epoch + 1, num_epochs),
                print numpy.mean(costs)
                #print profmode.print_summary()
                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'
        self.pro.terminate()
        self.pro2.terminate()
        self.pro3.terminate()
        self.pro4.terminate()
        self.pro5.terminate()
        self.pro6.terminate()
        self.pro7.terminate()
        self.pro8.terminate()
        self.weightQueue.close()
        self.weightQueue2.close()
        self.weightQueue3.close()
        self.weightQueue4.close()
        self.weightQueue5.close()
        self.weightQueue6.close()
        self.weightQueue7.close()
        self.weightQueue8.close()
        #print "NOW FETCHING",self.weightQueue.qsize()
        p2 = self.args[0]['params']
        #print "AFTER",len(p2)
        print p2[0]==self.params[0]
        self.W.set_value(p2[0])
        self.bv.set_value(p2[1])
        self.bh.set_value(p2[2])
        self.Wuh.set_value(p2[3])
        self.Wuv.set_value(p2[4])
        self.Wvu.set_value(p2[5])
        self.Wuu.set_value(p2[6])
        self.bu.set_value(p2[7])
    def generate(self, filename, show=True):
        '''Generate a sample sequence, plot the resulting piano-roll and save
it as a MIDI file.

filename : string
  A MIDI file will be created at this location.
show : boolean
  If True, a piano-roll of the generated sequence will be shown.'''

        piano_roll = self.generate_function()
        midiwrite(filename, piano_roll, self.r, self.dt)
        if show:
            extent = (0, self.dt * len(piano_roll)) + self.r
            pylab.figure()
            pylab.imshow(piano_roll.T, origin='lower', aspect='auto',
                         interpolation='nearest', cmap=pylab.cm.gray_r,
                         extent=extent)
            pylab.xlabel('time (s)')
            pylab.ylabel('MIDI note number')
            pylab.title('generated piano-roll')


def test_rnnrbm(batch_size=100, num_epochs=200):
    model = RnnRbm()
    #model.generate("test.mid",False)
    #re = os.path.join(os.path.split(os.path.dirname(__file__))[0],
    #                  'data', 'Nottingham', 'train', '*.mid')
    re = os.path.join(os.path.split(os.path.dirname(__file__))[0],
                      'data', 'Nottingham', 'train', '*.mid')
    #print re
    #for i in glob.glob(re):
    #  print i.split('/')[-1]
    #  transpose(converter.parse(i),i.split('/')[-1])
    #re = os.path.join(os.path.split(os.path.dirname(__file__))[0],
    #                  'data', 'converted', '*.mid')
    model.train(glob.glob(re),
                batch_size=batch_size, num_epochs=num_epochs)
    return model


def runner(n_visible,n_hidden,n_hidden_recurrent):
    W = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)
    #print params
    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu, -1000  # learned parameters as shared
    #print params
    return params

# params are the weights of the rnnrbm
# que is the stuff we need to process
# outque is the cost that is being passed out, along with test cost
# stop is flag meant to this process function to stop

def test_fn(queue):
   print "HI"
   queue.put(14)
if __name__ == '__main__':
    model = test_rnnrbm(100,200)
    #model.train()
    model.generate('long100sample1.mid',False)
    model.generate('long100sample2.mid',False)
    #pylab.show()
    print "DONE"
