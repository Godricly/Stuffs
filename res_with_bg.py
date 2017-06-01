#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import mxnet as mx
from iters import WordImageIter, TrueImageIter, TwoIter
from util import visual, WGANMetric

def res_module(data, nf, prefix, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    with mx.name.Prefix(prefix):
        conv = mx.sym.Convolution(data, name='conv', kernel=(3,3), pad=(1,1), num_filter=nf, no_bias=no_bias)
        bn = mx.sym.BatchNorm(conv, name='bn', fix_gamma=fix_gamma, eps=eps)
        act = mx.sym.Activation(bn, name='act', act_type='relu')
        return act + data

def conv_block(data, nf, prefix, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    with mx.name.Prefix(prefix):
        conv = mx.sym.Convolution(data, name='conv', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nf, no_bias=no_bias)
        bn = mx.sym.BatchNorm(conv, name='bn', fix_gamma=fix_gamma, eps=eps)
        act = mx.sym.Activation(bn, name='act', act_type='relu')
        return act

def deconv_block(data, nf, prefix, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    with mx.name.Prefix(prefix):
        deconv = mx.sym.Deconvolution(data, name='deconv', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf, no_bias=no_bias)
        bn = mx.sym.BatchNorm(deconv, name='bn', fix_gamma=fix_gamma, eps=eps)
        act = mx.sym.Activation(bn, name='act', act_type='relu')
        return act

def toreal(ndf, ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    data = mx.sym.Variable('data')
    data_d1 = conv_block(data, ndf, 'data_d1')
    data_d2 = conv_block(data_d1, ndf*2, 'data_d2')
    data_d3 = conv_block(data_d2, ndf*4, 'data_d3')
    data_res1 = res_module(data_d3, ndf*4, 'data_res1')
    data_res2 = res_module(data_res1, ndf*4, 'data_res2')
    data_res3 = res_module(data_res2, ndf*4, 'data_res3')

    bg = mx.sym.Variable('bg')
    bg_d1 = conv_block(bg, ndf, 'bg_d1')
    bg_d2 = conv_block(bg_d1, ndf*2, 'bg_d2')
    bg_d3 = conv_block(bg_d2, ndf*4, 'bg_d3')
    bg_res1 = res_module(bg_d3, ndf*4, 'bg_res1')
    bg_res2 = res_module(bg_res1, ndf*4, 'bg_res2')
    bg_res3 = res_module(bg_res2, ndf*4, 'bg_res3')

    feature = mx.sym.Concat(data_res3, bg_res3)

    #generate
    gen_res1 = res_module(feature, ndf*8, 'gen_res1')
    gen_res2 = res_module(gen_res1, ndf*8, 'gen_res2')
    gen_res3 = res_module(gen_res2, ndf*8, 'gen_res3')

    conv = mx.sym.Convolution(gen_res3, name='conv', kernel=(3,3), pad=(1,1), num_filter=ndf*4, no_bias=no_bias)
    bn = mx.sym.BatchNorm(conv, name='bn', fix_gamma=fix_gamma, eps=eps)
    act = mx.sym.Activation(bn, name='act', act_type='relu')

    g4 = mx.sym.Deconvolution(bn, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2, no_bias=no_bias)
    gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf, no_bias=no_bias)
    gbn5 = mx.sym.BatchNorm(g5, name='gbn5', fix_gamma=fix_gamma, eps=eps)
    gact5 = mx.sym.Activation(gbn5, name='gact5', act_type='relu')

    g6 = mx.sym.Deconvolution(gact5, name='g6', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g6, name='gact6', act_type='tanh')

    return gout

def tocharbg(ndf, ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    data = mx.sym.Variable('data')
    data_d1 = conv_block(data, ndf, 'data_d1')
    data_d2 = conv_block(data_d1, ndf*2, 'data_d2')
    data_d3 = conv_block(data_d2, ndf*4, 'data_d3')
    conv = mx.sym.Convolution(data_d3, name='conv', kernel=(3,3), pad=(1,1), num_filter=ndf*8, no_bias=no_bias)
    bn = mx.sym.BatchNorm(conv, name='bn', fix_gamma=fix_gamma, eps=eps)
    act = mx.sym.Activation(bn, name='act', act_type='relu')
    data_res1 = res_module(act, ndf*8, 'data_res1')
    data_res2 = res_module(data_res1, ndf*8, 'data_res2')
    data_res3 = res_module(data_res2, ndf*8, 'data_res3')
    [fg, bg] = mx.sym.split(data_res3, num_outputs=2)

    #gen fg
    gen_fg_res1 = res_module(fg, ndf*4, 'gen_fg_res1')
    gen_fg_res2 = res_module(gen_fg_res1, ndf*4, 'gen_fg_res2')
    gen_fg_res3 = res_module(gen_fg_res2, ndf*4, 'gen_fg_res3')
    fg_g4 = deconv_block(gen_fg_res3, ngf*2, 'fg_g4')
    fg_g5 = deconv_block(fg_g4, ngf*1, 'fg_g5')

    gen_bg_res1 = res_module(bg, ndf*4, 'gen_bg_res1')
    gen_bg_res2 = res_module(gen_bg_res1, ndf*4, 'gen_bg_res2')
    gen_bg_res3 = res_module(gen_bg_res2, ndf*4, 'gen_bg_res3')   
    bg_g4 = deconv_block(gen_bg_res3, ngf*2, 'bg_g4')
    bg_g5 = deconv_block(bg_g4, ngf*1, 'bg_g5')

    fg_g6 = mx.sym.Deconvolution(fg_g5, name='fg_g6', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias)
    fg_gout = mx.sym.Activation(fg_g6, name='fg_gact6', act_type='tanh')

    bg_g6 = mx.sym.Deconvolution(bg_g5, name='bg_g6', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias)
    bg_gout = mx.sym.Activation(bg_g6, name='bg_gact6', act_type='tanh')

    return mx.sym.Group([fg_gout, bg_gout])


def critic(ndf, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    data = mx.sym.Variable('data')
    BatchNorm = mx.sym.BatchNorm
    d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf, no_bias=no_bias)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4, no_bias=no_bias)
    dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8, no_bias=no_bias)
    dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8, no_bias=no_bias)
    dbn5 = BatchNorm(d5, name='dbn5', fix_gamma=fix_gamma, eps=eps)
    dact5 = mx.sym.LeakyReLU(dbn5, name='dact5', act_type='leaky', slope=0.2)

    d6 = mx.sym.Convolution(dact5, name='d6', kernel=(4,4), num_filter=1, no_bias=no_bias)
    d6 = mx.sym.Flatten(d6)

    return d6

if __name__ == '__main__':
    ctxs = [mx.gpu(1)]
    lr_d = 0.001
    lr_g = 0.001
    clip_weights = 0.01
    ndf = 32
    ngf = 32
    nc = 3
    epochs = 500
    log_step = 10
    batch_size= 128
    model_prefix = 'font_gan'
    m_d_real = WGANMetric()
    m_d_fg = WGANMetric()
    m_d_bg = WGANMetric()
    m_g_real = WGANMetric()
    m_g_fg = WGANMetric()
    m_g_bg = WGANMetric()

    words = TrueImageIter('train_full.txt', batch_size=batch_size, data_shape=(3,128,128))
    bg =  TrueImageIter('bg_list.txt', batch_size=batch_size, data_shape=(3,128,128))
    twoiter = TwoIter([words, bg])

    imgiter = TrueImageIter('clean_list.txt', batch_size=batch_size, data_shape=(3,128,128))


    with mx.name.Prefix('trans'):
        sym_toreal = toreal(ndf, ngf, nc)
        im_critic = critic(64)

    with mx.name.Prefix('inv'):
        sym_tocharbg = tocharbg(ndf, ngf, nc)
        char_critic = critic(64)
        bg_critic = critic(64)


    im_d = mx.mod.Module(symbol=im_critic, data_names=('data',), label_names=None, context=ctxs)
    im_d.bind(data_shapes=imgiter.provide_data,
              inputs_need_grad=True)
    im_d.init_params(initializer=mx.init.Normal(0.02))
    im_d.init_optimizer(
        optimizer='rmsprop',
        optimizer_params={
            'learning_rate': lr_d,
            'clip_weights': clip_weights,
        })

    char_d = mx.mod.Module(symbol=char_critic, data_names=('data',), label_names=None, context=ctxs)
    char_d.bind(data_shapes=imgiter.provide_data,
              inputs_need_grad=True)
    char_d.init_params(initializer=mx.init.Normal(0.02))
    char_d.init_optimizer(
        optimizer='rmsprop',
        optimizer_params={
            'learning_rate': lr_d,
            'clip_weights': clip_weights,
        })

    bg_d = mx.mod.Module(symbol=bg_critic, data_names=('data',), label_names=None, context=ctxs)
    bg_d.bind(data_shapes=imgiter.provide_data,
              inputs_need_grad=True)
    bg_d.init_params(initializer=mx.init.Normal(0.02))
    bg_d.init_optimizer(
        optimizer='rmsprop',
        optimizer_params={
            'learning_rate': lr_d,
            'clip_weights': clip_weights,
        })

    im_g = mx.mod.Module(symbol=sym_toreal, data_names=('data','bg'), label_names=None, context=ctxs)
    im_g.bind(data_shapes=twoiter.provide_data,
              inputs_need_grad=True)
    im_g.init_params(initializer=mx.init.Normal(0.02))
    im_g.init_optimizer(
        optimizer='rmsprop',
        optimizer_params={
            'learning_rate': lr_d,
        })



    char_g = mx.mod.Module(symbol=sym_tocharbg, data_names=('data',), label_names=None, context=ctxs)
    char_g.bind(data_shapes=imgiter.provide_data,
              inputs_need_grad=True)
    char_g.init_params(initializer=mx.init.Normal(0.02))
    char_g.init_optimizer(
        optimizer='rmsprop',
        optimizer_params={
            'learning_rate': lr_d,
        })

    # training
    for epoch in range(epochs):
        imgiter.reset()
        m_d_real.reset()
        m_d_fg.reset()
        m_d_bg.reset()
        m_g_real.reset()
        m_g_fg.reset()
        m_g_bg.reset()
        for idx, im in enumerate(imgiter):
            fg_bg = twoiter.next()
            im_g.forward(fg_bg)
            char_g.forward(im)
            #critiquing fake
            im_d.forward(mx.io.DataBatch(im_g.get_outputs(), []), is_train=True)
            char_d.forward(mx.io.DataBatch([char_g.get_outputs()[0]], []), is_train=True)
            bg_d.forward(mx.io.DataBatch([char_g.get_outputs()[1]], []), is_train=True)
            im_d.backward([mx.nd.ones((batch_size, 1))/batch_size])
            char_d.backward([mx.nd.ones((batch_size, 1))/batch_size])
            bg_d.backward([mx.nd.ones((batch_size, 1))/batch_size])

            fg2real = [[grad.copyto(grad.context) for grad in grads] for grads in im_d._exec_group.grad_arrays]  
            fg2fg = [[grad.copyto(grad.context) for grad in grads] for grads in char_d._exec_group.grad_arrays]
            fg2bg = [[grad.copyto(grad.context) for grad in grads] for grads in bg_d._exec_group.grad_arrays] 
            fs_real = im_d.get_outputs()[0].asnumpy()
            fs_fg = char_d.get_outputs()[0].asnumpy()
            fs_bg = bg_d.get_outputs()[0].asnumpy()

            #####################################
            #recreating penalty
            #####################################
            # im_gened = [ i.copy() for i in im_g.get_outputs()]
            # char_gened = [ i.copy() for i in char_g.get_outputs()]
            # im_g.forward(mx.io.DataBatch(char_gened, []))
            # char_g.forward(mx.io.DataBatch(im_gened, []))

            # diff_im = im_g.get_outputs()[0].as_in_context(im.data[0].context)- im.data[0]
            # diff_char = char_g.get_outputs()[0].as_in_context(fg_bg.data[0].context)- fg_bg.data[0]
            # diff_bg = char_g.get_outputs()[1].as_in_context(fg_bg.data[1].context)- fg_bg.data[1]

            #critiquing real
            im_d.forward(im, is_train=True)
            char_d.forward(mx.io.DataBatch([fg_bg.data[0]], []), is_train=True)
            bg_d.forward(mx.io.DataBatch([fg_bg.data[1]], []), is_train=True)
            #crtiquing real backward
            im_d.backward([-mx.nd.ones((batch_size, 1))/batch_size])
            char_d.backward([-mx.nd.ones((batch_size, 1))/batch_size])
            bg_d.backward([-mx.nd.ones((batch_size, 1))/batch_size])
            rs_real = im_d.get_outputs()[0].asnumpy()
            rs_fg = char_d.get_outputs()[0].asnumpy()
            rs_bg = bg_d.get_outputs()[0].asnumpy()

                       #update error
            err_real = -(fs_real - rs_real)
            err_fg = -(fs_fg - rs_fg)
            err_bg = -(fs_bg - rs_bg)
            m_d_real.update(err_real.mean())
            m_d_fg.update(err_fg.mean())
            m_d_bg.update(err_bg.mean())

            # gradient combine
            for grads_real, grads_fake in zip(char_d._exec_group.grad_arrays, fg2fg):
                for grad_real, grad_fake in zip(grads_real, grads_fake):
                    grad_real += grad_fake
            char_d.update()
            for grads_real, grads_fake in zip(bg_d._exec_group.grad_arrays, fg2bg):
                for grad_real, grad_fake in zip(grads_real, grads_fake):
                    grad_real += grad_fake
            bg_d.update()
            for grads_real, grads_fake in zip(im_d._exec_group.grad_arrays, fg2real):
                for grad_real, grad_fake in zip(grads_real, grads_fake):
                    grad_real += grad_fake
            im_d.update()

            # update decoder
            im_g.forward(fg_bg, is_train=True)
            char_g.forward(im, is_train=True)

            im_d.forward(mx.io.DataBatch(im_g.get_outputs(), []), is_train=True)
            im_d.backward([-mx.nd.ones((batch_size, 1))/batch_size])
            im_g.backward(im_d.get_input_grads())
            im_g.update()
            char_d.forward(mx.io.DataBatch([char_g.get_outputs()[0]], []), is_train=True)
            char_d.backward([-mx.nd.ones((batch_size, 1))/batch_size])
            bg_d.forward(mx.io.DataBatch([char_g.get_outputs()[1]], []), is_train=True)
            bg_d.backward([-mx.nd.ones((batch_size, 1))/batch_size])
            char_g.backward(char_d.get_input_grads()+bg_d.get_input_grads())
            char_g.update()
            err_real = -im_d.get_outputs()[0]
            m_g_real.update(err_real.asnumpy().mean())
            err_fg = -char_d.get_outputs()[0]
            m_g_fg.update(err_fg.asnumpy().mean())
            err_bg = -bg_d.get_outputs()[0]
            m_g_bg.update(err_bg.asnumpy().mean())

            if (idx+1)%log_step == 0:
                print("epoch:", epoch+1, "iter:", idx+1,
                 "real_g: ", m_g_real.get(), "real_d: ", m_d_real.get(),
                 "fg_g: ", m_g_fg.get(), "fg_d: ", m_d_fg.get(),
                 "bg_g: ", m_g_bg.get(), "bg_d: ", m_d_bg.get())

            im_g.forward(fg_bg)
            char_g.forward(im)
            im_gened = [ i.copy() for i in im_g.get_outputs()]
            char_gened = [ i.copy() for i in char_g.get_outputs()]
            im_g.forward(mx.io.DataBatch(char_gened, [], provide_data = twoiter.provide_data), is_train=True)
            char_g.forward(mx.io.DataBatch(im_gened, []), is_train=True)
            diff_im =  mx.nd.sign(im_g.get_outputs()[0].as_in_context(im.data[0].context)- im.data[0])
            diff_fg =  mx.nd.sign(char_g.get_outputs()[0].as_in_context(fg_bg.data[0].context)- fg_bg.data[0])
            diff_bg =  mx.nd.sign(char_g.get_outputs()[1].as_in_context(fg_bg.data[1].context)- fg_bg.data[1])

            im_g.backward([diff_im])
            char_g.backward([diff_fg, diff_bg])
            im_g.update()
            char_g.update()

        im_g.forward(fg_bg)
        char_g.forward(im)
        out_real = im_g.get_outputs()[0].asnumpy()
        out_fg = char_g.get_outputs()[0].asnumpy()
        out_bg = char_g.get_outputs()[1].asnumpy()

        #recreating
        im_g.forward(mx.io.DataBatch(char_g.get_outputs(), [] ,provide_data = twoiter.provide_data))
        char_g.forward(mx.io.DataBatch([mx.nd.array(out_real)], []))
        rec_real = im_g.get_outputs()[0].asnumpy()
        rec_fg = char_g.get_outputs()[0].asnumpy()
        rec_bg = char_g.get_outputs()[1].asnumpy()

        canvas = visual('tmp/char-rand-%d.png'%(epoch+1), fg_bg.data[0].asnumpy())
        canvas = visual('tmp/bg-rand-%d.png'%(epoch+1), fg_bg.data[1].asnumpy())
        canvas = visual('tmp/im-rand-%d.png'%(epoch+1), im.data[0].asnumpy())

        canvas = visual('tmp/out_real-rand-%d.png'%(epoch+1), out_real)
        canvas = visual('tmp/out_fg-rand-%d.png'%(epoch+1), out_fg)
        canvas = visual('tmp/out_bg-rand-%d.png'%(epoch+1), out_bg)

        canvas = visual('tmp/rec_real-rand-%d.png'%(epoch+1), rec_real)
        canvas = visual('tmp/rec_fg-rand-%d.png'%(epoch+1), rec_fg)
        canvas = visual('tmp/rec_bg-rand-%d.png'%(epoch+1), rec_bg)