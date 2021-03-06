import os
import numpy as np
import tensorflow as tf
from ops import Linear, linear, weight

batch_size = 32

def C(M, length, k, beta):
    M_norm = tf.sqrt(tf.reduce_sum(tf.square(M), 1, keep_dims=True)) # N * 1
    k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), 1, keep_dims=True))
    M_dot_k = tf.matmul(M, tf.reshape(k, [-1, 1])) # N * 1
    position = tf.reshape(tf.to_float(tf.range(length)), [-1, 1]) / 100
    similarity = M_dot_k * position * beta / (M_norm * k_norm + 1e-3)
    return tf.nn.softmax(tf.transpose(similarity)) # 1 * N

def D(length, k, k_size, beta):
    k = mod(k, length)
    # print(k.get_shape())
    similarity = -beta * tf.square(tf.tile(tf.reshape(tf.to_float(tf.range(length)), [-1, 1, 1]), [1, batch_size, k_size]) - tf.reshape(k, [1,batch_size,1]))
    return tf.nn.softmax(tf.transpose(similarity, [1,2,0])) # k_size * length

def mod(r1, r2):
    return r1 - r1 // r2 * r2

class PTRCell(object):
    def __init__(self, M_shape_list, read_PTR_list, write_PTR_list, G_shape, controller_layer_size=2, controller_hidden_size=256, init_range=0.1, addr_mode=0):
        self.M_shapes = M_shape_list
        self.read_ptrs = read_PTR_list
        self.write_ptrs = write_PTR_list
        self.G_shape = G_shape
        self.ctl_layer_size = controller_layer_size
        self.ctl_hidden_size = controller_hidden_size
        self.init_range = init_range
        self.addr_mode = addr_mode
        
        self.ptr_size = sum(self.read_ptrs) + sum(self.write_ptrs)
        self.read_sizes = [self.read_ptrs[i] * self.M_shapes[i][2] for i in range(len(self.read_ptrs))]
        self.write_sizes = [self.write_ptrs[i] * self.M_shapes[i][2] for i in range(len(self.write_ptrs))]
        self.ctl_output_dim = sum(self.write_sizes) + self.G_shape[0]*self.G_shape[1] + self.M_shapes[0][1] + self.ptr_size + 2

    def init_state(self, dummy_value=0.0):
        dummy = tf.Variable(tf.zeros([batch_size,1], dtype=tf.float32))

        M = [tf.zeros(shape) for shape in self.M_shapes]
        R = tf.zeros([batch_size, self.G_shape[0] * self.G_shape[1]])
        val = tf.zeros([batch_size, sum(self.read_sizes)])

        # controller state
        output_init_list = []
        hidden_init_list = []
        for idx in range(self.ctl_layer_size):
            output_init_idx = Linear(dummy, self.ctl_hidden_size, squeeze=True, stddev=self.init_range, name='output_init_%s' % idx)
            output_init_list.append(tf.tanh(output_init_idx))
            hidden_init_idx = Linear(dummy, self.ctl_hidden_size, squeeze=True, stddev=self.init_range, name='hidden_init_%s' % idx)
            hidden_init_list.append(tf.tanh(hidden_init_idx))

        state = {
            'M': M,
            'R': R,
            'val': val,
            'ptr': tf.zeros([batch_size, self.ptr_size]), #weight('ptr_init', [1, self.ptr_size], init='constant'),
            'dptr': tf.zeros([batch_size, self.ptr_size]),
            'seq_length': 0,
            'output': output_init_list,
            'hidden': hidden_init_list,
        }

        self.__call__(state,0)
        return state

    def update_memory(self, state, X_list, limit=0.8):
         seq = state['seq_length']
         M = []
         if self.addr_mode == 0 or self.addr_mode == 1:
             for i in range(len(self.M_shapes)):
                 if self.M_shapes[i][1] > seq:
                     M.append(tf.concat([state['M'][i][:,0:seq, :], X_list[i], state['M'][i][:,seq+1:,:]], 1))
         #else:
             #similarity = C(state['M'][0], self.M_shapes[0][0], X_list[0], 1)
             #values, indices = tf.nn.top_k(similarity)
             #addr = tf.cond(values[0][0] > limit, lambda: D(self.M_shapes[0][0], tf.to_float(indices), 1, 100), lambda: D(self.M_shapes[0][0], tf.constant([[seq]], dtype=tf.float32), 1, 100))
             #addr = tf.cond(values[0][0] > limit, lambda: C(state['M'][0], self.M_shapes[0][0], X_list[0], 100), lambda: D(self.M_shapes[0][0], tf.constant([[seq]], dtype=tf.float32), 1, 100))
             #for i in range(len(self.M_shapes)):
             #    M.append(state['M'][i] * tf.transpose(1.0 - addr) + tf.matmul(tf.transpose(addr), X_list[i]))

         new_state = {
             'M': M,
             'R': state['R'],
             'val': state['val'],
             'ptr': state['ptr'],
             'dptr': state['dptr'],
             'seq_length': seq+1,
             'output': state['output'],
             'hidden': state['hidden'],
         }

         return new_state

    def __call__(self, state, steps):
        length_t = tf.constant([[state['seq_length'], steps]], dtype=tf.float32)
        ctl_in = tf.concat([state['val'], state['R']], 1)
        # state['dptr'] = tf.ones([1,self.ptr_size], dtype=tf.float32),  # dptr,
        # build a controller
        output_list, hidden_list  = self.build_LSTM_controller(ctl_in, state['output'], state['hidden'])

        ctl_out = tf.reshape(linear(output_list, self.ctl_output_dim, bias=True, stddev=self.init_range, scope='ctl_output'), [batch_size, -1])

        new_state = self.build_memory(ctl_out, state, output_list, hidden_list)
        # print(steps)
        
        return new_state



    def build_LSTM_controller(self, input_, output_list_prev, hidden_list_prev, scope="content"):
        """Build LSTM controller."""

        with tf.variable_scope("controller_" + scope):
            output_list = []
            hidden_list = []
            for layer_idx in range(self.ctl_layer_size):
                o_prev = output_list_prev[layer_idx]
                h_prev = hidden_list_prev[layer_idx]

                if layer_idx == 0:
                    def new_gate(gate_name):
                        return linear([input_, o_prev],
                                      output_size = self.ctl_hidden_size,
                                      bias = True,
                                      stddev=self.init_range, 
                                      scope = "%s_gate_%s" % (gate_name, layer_idx))
                else:
                    def new_gate(gate_name):
                        return linear([output_list[-1], o_prev],
                                      output_size = self.ctl_hidden_size,
                                      bias = True,
                                      stddev=self.init_range, 
                                      scope="%s_gate_%s" % (gate_name, layer_idx))

                # input, forget, and output gates for LSTM
                i = tf.sigmoid(new_gate('input'))
                f = tf.sigmoid(new_gate('forget'))
                o = tf.sigmoid(new_gate('output'))
                update = tf.tanh(new_gate('update'))

                # update the sate of the LSTM cell
                hid = tf.add_n([f * h_prev, i * update])
                out = o * tf.tanh(hid)

                hidden_list.append(hid)
                output_list.append(out)

            return output_list, hidden_list

    def build_memory(self, ctl_out, state, output_list, hidden_list):
        """Build a memory to read & write."""
        collected = {}
        with tf.variable_scope("memory"):
            # print('memory')
            idx = np.cumsum([0, sum(self.write_sizes), self.G_shape[0]*self.G_shape[1], self.M_shapes[0][2], self.ptr_size, 1, 1])
            Y = ctl_out[:, idx[0]:idx[1]]                                    # write  cr'
            R = ctl_out[:, idx[1]:idx[2]]                                    # G
            key = ctl_out[:, idx[2]:idx[3]]                                  # key for key,value M
            dptr = tf.nn.softplus(ctl_out[:, idx[3]:idx[4]])                 # 1 * self.ptr_size
            beta_r = 1+tf.nn.softplus(ctl_out[:, idx[4]:idx[5]])             # 1
            beta_w = 1+tf.nn.softplus(ctl_out[:, idx[5]:idx[6]])             # 1


            # dptr = tf.ones(tf.shape(dptr), dtype = tf.float32)

            M = []
            val = []
            if self.addr_mode == 0:
                ptr = state['ptr'] + state['dptr']
                read_ptr_start = 0
                write_ptr_start = sum(self.read_ptrs)
                write_start = 0
                collected['a_r'] = []
                collected['a_w'] = []
                for i in range(len(self.M_shapes)):
                    if self.read_ptrs[i] != 0:
                        a_r = D(self.M_shapes[i][1], ptr[:, read_ptr_start:read_ptr_start+self.read_ptrs[i]], self.read_ptrs[i], beta_r) # k * length
                        # print('jksdhvasgdysajfgyd', i, ' ',len(self.M_shapes) )
                        # print(a_r.get_shape(), state['M'][i].get_shape())
                        mulres = tf.einsum('aij,ajk->aik',a_r, state['M'][i])
                        val.append(tf.reshape(mulres, [batch_size, self.read_sizes[i]]))
                        read_ptr_start += self.read_ptrs[i]
                        collected['a_r'].append(a_r)

                    if self.write_ptrs[i] != 0:
                        a_w = D(self.M_shapes[i][1], state['ptr'][:, write_ptr_start:write_ptr_start+self.write_ptrs[i]], self.write_ptrs[i], beta_w) # k * length
                        c_w = tf.reshape(Y[:, write_start:write_start+self.write_sizes[i]], [batch_size, self.write_ptrs[i], self.M_shapes[i][2]])
                        M.append(state['M'][i] * tf.transpose(1.0 - tf.reduce_sum(a_w, 0, keep_dims=True), [0,2,1]) + tf.einsum('aij,ajk->aik',tf.transpose(a_w, [0,2,1]), c_w))
                        write_ptr_start += self.write_ptrs[i]
                        write_start += self.write_sizes[i]
                        collected['a_w'].append(a_w)
                    else:
                        M.append(state['M'][i])
                #collected['a_r'] = tf.concat(collected['a_r'], 0)
                collected['a_w'] = tf.concat(collected['a_w'], 0)
            elif self.addr_mode == 1:
                a = C(state['M'][0], self.M_shapes[0][0], key, beta_r) # 1 *  N
                write_start = 0
                for i in range(len(self.M_shapes)):
                    if self.read_ptrs[i] != 0:
                        val.append(tf.matmul(a, state['M'][i]))
                    if self.write_ptrs[i] != 0:
                        c_w = tf.reshape(Y[:, write_start:write_start+self.write_sizes[i]], [self.write_ptrs[i], self.M_shapes[i][1]])
                        M.append(state['M'][i] * tf.transpose(1.0 - tf.reduce_sum(a, 0, keep_dims=True)) + tf.matmul(tf.transpose(a), c_w))
                        write_start += self.write_sizes[i]
                    else:
                        M.append(state['M'][i])
                ptr = state['ptr']
                collected['a_r'] = a
                collected['a_w'] = a

            collected['beta_r'] = beta_r
            collected['beta_w'] = beta_w

            new_state = {
                        'M': M,
                        'R': R,
                        'val': tf.concat(val, 1),
                        'ptr': ptr,
                        'dptr': dptr,
                        'seq_length': state['seq_length'],
                        'output': output_list,
                        'hidden': hidden_list,
                        'collected': collected,
                    }
            return new_state
