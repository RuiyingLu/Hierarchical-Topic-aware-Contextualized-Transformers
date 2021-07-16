import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

def src_attn(x, phi, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_src_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        # w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a,w

    with tf.variable_scope(scope):
        q = conv1d(x, 'src_q_attn', n_state)
        k_v = conv1d(phi, 'src_kv_attn', n_state*2)
        q = split_heads(q)  # 1*1024*12*n_d
        if len(k_v.shape) == 3:
            k, v = map(split_heads, tf.split(k_v, 2, axis=2))
        else:
            k, v = map(split_heads, tf.split(tf.expand_dims(k_v,0), 2, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a,w = multihead_src_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present, w


def attn_att(topic, x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def mask_attn_att_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v, topic):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        topic_w = mask_attn_att_weights(tf.matmul(topic, topic, transpose_b=True))
        w = mask_attn_weights(w)
        w = w - tf.reduce_max(w, axis=-1, keepdims=True)
        att_weight = topic_w * tf.exp(w)
        w = att_weight / tf.reduce_sum(att_weight, axis=-1, keepdims=True)
        # w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v,topic)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

def attn_a_t(topic, x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def mask_attn_att_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v, topic):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        topic_w = mask_attn_att_weights(tf.matmul(topic, topic, transpose_b=True))
        w = mask_attn_weights(w)
        w = w - tf.reduce_max(w, axis=-1, keepdims=True)
        att_weight = topic_w * tf.exp(w)
        w = att_weight / tf.reduce_sum(att_weight, axis=-1, keepdims=True)
        # w = softmax(w)
        a = tf.matmul(w, v)
        return a

    def multihead_attn_a_t(q, k, v, topic):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        topic_w = mask_attn_att_weights(tf.matmul(topic, topic, transpose_b=True))
        w = mask_attn_weights(w)
        w = w - tf.reduce_max(w, axis=-1, keepdims=True)
        w = softmax(topic_w + w)
        # w = softmax(w)
        a = tf.matmul(w, v)
        return a


    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn_a_t(q, k, v,topic)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def block_topic_1(phi, x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        b, p, w = src_attn(norm(x, 'ln_3'), phi, 'attn', nx, past=past, hparams=hparams)
        x = x + b
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def block_topic_2(phi_attention,phi, x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        b, p, w = src_attn(norm(x, 'ln_3'), phi, 'attn', nx, past=past, hparams=hparams)
        x = x + b
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m

        attention_gt = tf.tile(tf.expand_dims(phi_attention, 1), [1, 12, 1, 1])
        att_loss = tf.nn.l2_loss(attention_gt-w)/hparams.n_head

        return x, present, att_loss

def block_topic_5(phi_attention,phi, x, scope, *, past, hparams):
    ## phi_attention: 1*1024*k
    ## phi: k*v
    ## x: 1*1024*embed
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)  ## self-atten
        x = x + a
        #phi_self_atten, _ = attn(norm(phi, 'ln_1'), 'attn', nx, past=past,hparams=hparams)
        b, p, w = src_attn(norm(x, 'ln_3'), phi, 'attn', nx, past=past, hparams=hparams)   ## word-phi atten
        x = x + b
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m

        attention_gt = tf.tile(tf.expand_dims(phi_attention, 1), [1, 12, 1, 1])
        att_l2_loss = tf.reduce_sum(tf.norm(attention_gt[:,:,1:,:] - w[:,:,-attention_gt.shape[2].value:-1,:],axis=-1))/hparams.n_head
        att_l2 = tf.norm(w,axis=-1,keep_dims=True)
        att_l1 = tf.norm(tf.div(w,att_l2),ord=1,axis=-1)
        att_sparse_loss = tf.reduce_sum(att_l1)/hparams.n_ctx
        att_loss = att_sparse_loss + att_l2_loss

        return x, present, att_loss, w


def block_topic_5_phi_atten(phi_attention,phi, x, scope, *, past, hparams):
    ## phi_attention: 1*1024*k
    ## phi: k*v
    ## x: 1*1024*embed
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)  ## self-atten
        x = x + a
        phi_self_atten, _ = attn(norm(tf.expand_dims(phi,0), 'ln_1'), 'attn', nx, past=past,hparams=hparams)
        b, p, w = src_attn(norm(x, 'ln_3'), phi_self_atten, 'attn', nx, past=past, hparams=hparams)   ## word-phi atten
        x = x + b
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m

        attention_gt = tf.tile(tf.expand_dims(phi_attention, 1), [1, 12, 1, 1])
        att_l2_loss = tf.reduce_sum(tf.norm(attention_gt[:,:,1:,:] - w[:,:,-attention_gt.shape[2].value:-1,:],axis=-1))/hparams.n_head
        att_l2 = tf.norm(w,axis=-1,keep_dims=True)
        att_l1 = tf.norm(tf.div(w,att_l2),ord=1,axis=-1)
        att_sparse_loss = tf.reduce_sum(att_l1)/hparams.n_ctx
        att_loss = att_sparse_loss + att_l2_loss

        return x, present, att_loss, w



def block_att(topic, x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn_att(topic, norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results


def model_layer3_att(hparams, X, Phi,Theta, past=None, scope='model', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))  ## 1*1024*n_emd

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        phi_1 = tf.matmul(Phi[1], Phi[0])
        phi_2 = tf.matmul(Phi[2], phi_1)
        phi_v_0 = tf.gather(tf.transpose(Phi[0]), X)*tf.transpose(Theta[0],[1,2,0])  ## 1*1024*k
        phi_v_1 = tf.gather(tf.transpose(phi_1), X)*tf.transpose(Theta[1],[1,2,0])
        phi_v_2 = tf.gather(tf.transpose(phi_2), X)*tf.transpose(Theta[2],[1,2,0])

        phi_v_0_s = phi_v_0 / tf.reduce_sum(phi_v_0, axis=-1, keep_dims=True)
        phi_v_1_s = phi_v_1 / tf.reduce_sum(phi_v_1, axis=-1, keep_dims=True)
        phi_v_2_s = phi_v_2 / tf.reduce_sum(phi_v_2, axis=-1, keep_dims=True)
        att_loss_all = 0

        for layer, past in enumerate(pasts):
            # h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 3:
                h, present, att_loss, att_1 = block_topic_5(phi_v_0_s, Phi[0], h, 'h%d' % layer, past=past,
                                                            hparams=hparams)
                att_loss_all += att_loss
            elif layer == 7:
                h, present, att_loss, att_2 = block_topic_5(phi_v_1_s, Phi[1], h, 'h%d' % layer, past=past,
                                                            hparams=hparams)
                att_loss_all += att_loss
            elif layer == 11:
                h, present, att_loss, att_3 = block_topic_5(phi_v_2_s, Phi[2], h, 'h%d' % layer, past=past,
                                                            hparams=hparams)
                att_loss_all += att_loss
            else:
                h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)

            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        results['att_loss'] = att_loss_all
        results['att_1'] = att_1
        results['att_2'] = att_2
        results['att_3'] = att_3

        return results

def model_layer3_embed(hparams, X, topic, thetasize, past=None, scope='model', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        w_topic = tf.get_variable('w_topic', [thetasize, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        topic_embedding = tf.reshape(tf.matmul(topic, w_topic),[batch,sequence,hparams.n_embd])
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length)) + topic_embedding

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results

def model_layer3_virtual(hparams, X, Theta, Phi, past=None, scope='model', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        w_topic_1 = tf.get_variable('w_topic_1',
                             initializer= wte)
        w_topic_2 = tf.get_variable('w_topic_2',
                             initializer=tf.matmul(Phi[0], wte))
        w_topic_3 = tf.get_variable('w_topic_3',
                             initializer=tf.matmul(Phi[1], tf.matmul(Phi[0], wte)))
        phi_num1 = Phi[0].shape[0]
        phi_num2 = Phi[1].shape[0]
        phi_num3 = Phi[2].shape[0]
        # w_topic_1 = tf.get_variable('w_topic_1', [hparams.n_vocab, hparams.n_embd],
        #                           initializer=tf.random_normal_initializer(stddev=0.01))
        # w_topic_2 = tf.get_variable('w_topic_2', [phi_num1, hparams.n_embd],
        #                           initializer=tf.random_normal_initializer(stddev=0.01))
        # w_topic_3 = tf.get_variable('w_topic_3', [phi_num2, hparams.n_embd],
        #                           initializer=tf.random_normal_initializer(stddev=0.01))
        topic1=tf.einsum('ibn,nd->ibd', Phi[0]*tf.reshape(tf.transpose(Theta[0][:,:,0]),[batch,phi_num1,1]),w_topic_1)
        topic2=tf.einsum('ibn,nd->ibd', Phi[1]*tf.reshape(tf.transpose(Theta[1][:,:,0]),[batch,phi_num2,1]),w_topic_2)
        topic3=tf.einsum('ibn,nd->ibd', Phi[2]*tf.reshape(tf.transpose(Theta[2][:,:,0]),[batch,phi_num3,1]),w_topic_3)
        virtual_topic = tf.concat([topic1,topic2,topic3],axis=1) # batchsize * topic num * embedding size  1*230*768

        h = tf.concat([virtual_topic,h],axis=1)
        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h[:,-sequence:,:], [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results


