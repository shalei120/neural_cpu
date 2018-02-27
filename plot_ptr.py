import matplotlib.pyplot as plt
import numpy as np
import pickle, sys

np.set_printoptions(suppress=True, precision=1, linewidth=200)

def pprint(seq):
    seq = np.array(seq)
    dim = 1
    seq = np.char.mod('%d', np.around(seq))
    seq[:, (dim, dim+2, dim+4)] = ' '
    print("\n".join(["".join(x) for x in seq.tolist()]))

def plot_addr(seq1, seq2, true, out, states):
    seq_length = len(true)
    #for i in range(seq_length*2):
        #plt.plot(np.arange(20),np.reshape(states[i]['w_w'], [-1])+i*3, '-b')
        #plt.plot(np.arange(20),np.reshape(states[i]['read_w'], [-1])+i*3, '--b')
        #plt.plot(np.arange(20), np.reshape(states[i]['collected']['a_w'], [-1])+i*3, '--b')
        #collect = states[i]['collected']
        #print("g_addr:%.1f X:%r" % (collect['g_addr'], collect['X']))

    Read = 2
    Write = 1
    total = 2
    #Read = 2
    #Write = 1
    #total = Read + 1
    
    print (np.hstack(
            [np.reshape(np.stack(states[seq_length-1]['M'], axis=0)[:,0:seq_length, :], [total*seq_length,10+5]),
             np.ones([total*seq_length, 2]),
             np.reshape(np.stack(states[-1]['M'], axis=0)[:, 0:seq_length, :], [total*seq_length,10+5])]
    ))
    read_start = seq_length
    for i in range(read_start, len(states)):
        collect = states[i]['collected']
        for j in range(Read+Write):
            if j < Read:
                vec = collect['a_r'][j,:]
                color = '-g'
            else:
                vec = collect['a_w'][j-Read,:]
                color = '-b'
            plt.plot(np.arange(vec.shape[-1]), np.reshape(vec, [-1])+(i-read_start)*5+j, color)
        
        #for j in range(Read+Write):
        #    vec = collect['a_r'][j,:]
        #    plt.plot(np.arange(vec.shape[-1]), np.reshape(vec, [-1])+(i-read_start)*5+j+2, '-g')
        #print("old_ptr:%s ptr:%s dptr:%s b_r:%.1f b_w:%.1f" % (states[i]['old_ptr'], states[i]['ptr'], collect['dptr'], collect['beta_r'], collect['beta_w']))
        print("ptr:%s dptr:%s b_r:%.1f b_w:%.1f" % (states[i]['ptr'], states[i]['dptr'], collect['beta_r'], collect['beta_w']))
    plt.show()

    pprint(np.hstack([np.asarray(seq1), np.zeros((seq_length, 1)), np.asarray(seq2), np.zeros((seq_length, 1)), np.asarray(true), np.zeros((seq_length, 1)), np.asarray(out)]))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage: python plot_data.py [test_case]")
        exit(1)

    f = open(sys.argv[1], 'rb')
    seq1, seq2, true, out, states = pickle.load(f)
    plot_addr(seq1, seq2, true, out, states)   
