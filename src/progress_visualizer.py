def to_console(logger, name, epoch, it, losses, performances, other_metrics=None, n_iters_per_epoch=1, lr=None):
    progress_str = "\n" + "-" * 60 + "\n"
    progress_str += f"[{name}] Epoch {epoch}, Iteration {it + 1}, Learning Rate {lr}\n"
    progress_str += "." * 30 + "\n"
    progress_str += "\n".join([f"{n:16s}: {l.item():.4f}" for n, l in losses.items()]) + "\n\n"
    progress_str += "\n".join([f"{n:16s}: {p.item():.4f}" for n, p in performances.items()])
    if other_metrics:
        progress_str += "\n\n" + "\n".join([f"{n:16s}: {m.item():.4f}" for n, m in other_metrics.items()])
    logger.info(progress_str)


def to_tensorboard(tb_writer, name, epoch, it, losses, performances, other_metrics=None, n_iters_per_epoch=1, lr=None):
    step = epoch * n_iters_per_epoch + it
    for n, l in losses.items():
        tb_writer.add_scalar(f"loss/{name}_{n}", l.item(), step)
    for n, p in performances.items():
        tb_writer.add_scalar(f"performances/{name}_{n}", p.item(), step)
    if other_metrics:
        for n, m in other_metrics.items():
            tb_writer.add_scalar(f"other_metrics/{name}_{n}", m.item(), step)


class ProgressVisualizer:
    def __init__(self, name, n_iters_per_epoch=1, show_progress_every_n_iters=1, tb_writer=None, console_logger=None):
        self.name = name
        self.n_iters_per_epoch = n_iters_per_epoch
        self.show_progress_every_n_iters = show_progress_every_n_iters
        self.console_logger = console_logger
        self.tb_writer = tb_writer
        self.display_funcs = []
        self.handlers = []

        if self.tb_writer is not None:
            self.display_funcs.append(to_tensorboard)
            self.handlers.append(self.tb_writer)
        if self.console_logger is not None:
            self.display_funcs.append(to_console)
            self.handlers.append(self.console_logger)

    def display(self, epoch, it, losses, performances, other_metrics=None, lr=None):
        if (it + 1) % self.show_progress_every_n_iters == 0:
            for display_func, handler in zip(self.display_funcs, self.handlers):
                display_func(
                    handler,
                    self.name,
                    epoch,
                    it,
                    losses,
                    performances,
                    other_metrics=other_metrics,
                    n_iters_per_epoch=self.n_iters_per_epoch,
                    lr=lr,
                )
