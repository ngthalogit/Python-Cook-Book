
# get learning rate
def get_lr(opt):
    for param_gr in opt.param_group:
        return param_gr['lr']

def loss_batch(loss_func, y_batch, y_hat, opt=None):
    loss = loss_func(y_hat, y_batch)
    pred = y_hat.argmax(dim=1, keepdims=True)
    corrects = pred.eq(y_batch.view_as(y_hat)).sum().item()
    if opt is not None:
        loss.requires_grad_(True)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), corrects


