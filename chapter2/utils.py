from matplotlib import pyplot as plt
# get learning rate
def get_lr(opt):
    for param_gr in opt.param_groups:
        return param_gr['lr']

def loss_batch(loss_func, y_batch, y_hat, opt=None):
    loss = loss_func(y_hat, y_batch)
    pred = y_hat.argmax(dim=1, keepdims=True)
    corrects = pred.eq(y_batch.view_as(pred)).sum().item()
    if opt is not None:
        loss.requires_grad_(True)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), corrects

def plot_process(num_epoches, loss_hist, metric_hist):
    plt.title('train_val_loss')
    plt.plot(range(1, num_epoches+1), loss_hist['train'], labels='train')
    plt.plot(range(1, num_epoches+1), loss_hist['val'], labels='val')
    plt.ylabel('loss')
    plt.xlabel('epoches')
    plt.legend()
    plt.show()

    plt.title('train_val_accuracy')
    plt.plot(range(1, num_epoches+1), metric_hist['train'], labels='train')
    plt.plot(range(1, num_epoches+1), metric_hist['val'], labels='val')
    plt.ylabel('accuracy')
    plt.xlabel('epoches')
    plt.legend()
    plt.show()

