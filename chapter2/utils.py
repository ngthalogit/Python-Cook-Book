
# get learning rate
def get_lr(opt):
    for param_gr in opt.param_group:
        return param_gr['lr']

def loss_batch(loss_func, y_batch, y_hat):
    loss_b = loss_func(y_hat,y_batch)
