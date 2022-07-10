import torch

class EarlyStopping():
    def __init__(self, patience=5, save_path=None):
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, net, opt, epoch, val_losses, train_losses):

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, net, opt, epoch, val_losses, train_losses)
        elif val_loss > self.best_score:
            self.counter += 1
            print('Validation Loss has not decreased [{}/{}]'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, net, opt, epoch, val_losses, train_losses)
            self.counter = 0

    def save_checkpoint(self, val_loss, net, opt, epoch, val_losses, train_losses):
        """Save model as checkpoint if validation loss decreased"""
        torch.save({
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'opt_state_dict': opt.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            'val_losses': val_losses,
            'train_losses': train_losses,
        }, self.save_path)
        print('Checkpoint saved')

