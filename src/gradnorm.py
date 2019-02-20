import torch
import torch.nn as nn


def compute_grad_l2_norm(layers, loss):
    G = torch.autograd.grad(loss, layers, retain_graph=True, create_graph=True)
    G_norm = torch.cat([torch.norm(g, 2).unsqueeze(0) for g in G]).sum()
    return G_norm


class SimpleGradNormalizer:
    def __init__(self, lr_init=0.025, alpha=0.16):
        self.alpha = alpha
        self.init_loss = None
        self.loss_weight = nn.Parameter(torch.ones(3))
        # self.loss_weight = [torch.tensor(1., requires_grad=True) for _ in range(3)]
        # self.w1 = nn.Parameter(torch.tensor([1.]))
        # self.w2 = nn.Parameter(torch.tensor([1.]))
        # self.w3 = nn.Parameter(torch.tensor([1.]))
        self.optim = torch.optim.Adam([self.loss_weight], lr=lr_init)

    def set_init_loss(self, losses):
        self.init_loss = {n: torch.tensor([l.item()], device=l.device) for n, l in losses.items()}

    def normalize_loss_weight(self):
        num_losses = len(self.init_loss)
        coef = num_losses / self.loss_weight.sum()
        self.loss_weight.data[:] = self.loss_weight * coef

    def adjust_losses(self, losses):
        if self.init_loss is None:
            self.set_init_loss(losses)
        # losses["cls_loss"] *= self.loss_weight[0]
        # losses["bbox_loss"] *= self.loss_weight[1]
        # losses["counter_loss"] *= self.loss_weight[2]
        losses["cls_loss"] = losses["cls_loss"] * self.loss_weight[0]
        losses["bbox_loss"] = losses["bbox_loss"] * self.loss_weight[1]
        losses["counter_loss"] = losses["counter_loss"] * self.loss_weight[2]

    def adjust_grad(self, losses, model):
        losses["total_loss"].backward(retain_graph=True)

        shared_layers = [
            model.layer1.conv.weight,
            model.layer1.conv.bias,
            model.layer2_2.conv.weight,
            model.layer2_2.conv.bias,
        ]

        G_cls_norm = compute_grad_l2_norm(shared_layers, losses["cls_loss"])
        G_loc_norm = compute_grad_l2_norm(shared_layers, losses["bbox_loss"])
        G_cnt_norm = compute_grad_l2_norm(shared_layers, losses["counter_loss"])

        G_avg = (G_cls_norm + G_loc_norm + G_cnt_norm) / 3

        lhat_cls = losses["cls_loss"] / self.init_loss["cls_loss"]
        lhat_loc = losses["bbox_loss"] / self.init_loss["bbox_loss"]
        lhat_cnt = losses["counter_loss"] / self.init_loss["counter_loss"]
        lhat_avg = (lhat_cls + lhat_loc + lhat_cnt) / 3

        inv_rate_cls = lhat_cls / lhat_avg
        inv_rate_loc = lhat_loc / lhat_avg
        inv_rate_cnt = lhat_cnt / lhat_avg

        C_cls = (G_avg * (inv_rate_cls) ** self.alpha).detach()
        C_loc = (G_avg * (inv_rate_loc) ** self.alpha).detach()
        C_cnt = (G_avg * (inv_rate_cnt) ** self.alpha).detach()

        print(f"=" * 120)
        print(f"|   G   | {G_cls_norm.item()}, {G_loc_norm.item()}, {G_cnt_norm.item()}")
        print(f"| lhat  | {lhat_cls.item()}, {lhat_loc.item()}, {lhat_cnt.item()}")
        print(f"| inv_r | {inv_rate_cls.item()}, {inv_rate_loc.item()}, {inv_rate_cnt.item()}")
        print(f"|   C   | {C_cls.item()}, {C_loc.item()}, {C_cnt.item()}")
        print(f"=" * 120)

        self.optim.zero_grad()
        Lgrad = nn.L1Loss()(G_cls_norm, C_cls) + nn.L1Loss()(G_loc_norm, C_loc) + nn.L1Loss()(G_cnt_norm, C_cnt)
        Lgrad.backward(retain_graph=True)
        self.optim.step()

        self.normalize_loss_weight()


class ComplexGradNormalizer:
    def __init__(self, lr_init=0.025, alpha=0.16):
        self.alpha = alpha
        self.init_loss = None
        self.sharing_info = {
            "shared_layers": [
                "layer1.conv.weight",
                "layer1.conv.bias",
                "layer2_2.conv.weight",
                "layer2_2.conv.bias",
            ],  # actually there're more shared layers, but we only choose the most representative ones.
            "losses": [
                {
                    "shared_layers": [
                        "layer1.conv.weight",
                        "layer1.conv.bias",
                        "layer2_2.conv.weight",
                        "layer2_2.conv.bias",
                        "layer3_3.conv.weight",
                        "layer3_3.conv.bias",
                    ],
                    "losses": [{"name": "cls_loss"}, {"name": "bbox_loss"}],
                },
                {"name": "counter_loss"},
            ],
        }
        self.weights_cls_vs_loc = torch.ones(2, requires_grad=True)
        self.optim_cls_vs_loc = torch.optim.Adam([self.weights_cls_vs_loc], lr=lr_init)
        self.weights_clsloc_vs_counter = torch.ones(2, requires_grad=True)
        self.optim_clsloc_vs_counter = torch.optim.Adam([self.weights_clsloc_vs_counter], lr=lr_init)

    def set_init_loss(self, losses):
        self.init_loss = {n: torch.tensor([l.item()], device=l.device) for n, l in losses.items()}

    def adjust_losses(self, losses):
        if self.init_loss is None:
            self.set_init_loss(losses)
        losses["cls_loss"] *= self.weights_cls_vs_loc[0] * self.weights_clsloc_vs_counter[0]
        losses["bbox_loss"] *= self.weights_cls_vs_loc[1] * self.weights_clsloc_vs_counter[0]
        losses["counter_loss"] *= self.weights_clsloc_vs_counter[1]

    def adjust_grad(self, losses, model):
        losses["total_loss"].backward(retain_graph=True)

        cls_loc_shared_layers = [
            model.layer1.conv.weight,
            model.layer1.conv.bias,
            model.layer2_2.conv.weight,
            model.layer2_2.conv.bias,
            model.layer3_3.conv.weight,
            model.layer3_3.conv.bias,
        ]
        G_cls_norm = compute_grad_l2_norm(cls_loc_shared_layers, losses["cls_loss"])
        G_loc_norm = compute_grad_l2_norm(cls_loc_shared_layers, losses["bbox_loss"])
        G_clsloc_avg = (G_cls_norm + G_loc_norm) / 2

        lhat_cls = losses["cls_loss"] / self.init_loss["cls_loss"]
        lhat_loc = losses["bbox_loss"] / self.init_loss["bbox_loss"]

        lhat_avg = (lhat_cls + lhat_loc) / 2

        inv_rate1 = lhat_cls / lhat_avg
        inv_rate2 = lhat_loc / lhat_avg

        C1 = (G_clsloc_avg * (inv_rate1) ** self.alpha).detach()
        C2 = (G_clsloc_avg * (inv_rate2) ** self.alpha).detach()

        self.optim_cls_vs_loc.zero_grad()
        L_clsloc = nn.L1Loss()(torch.cat(G_cls_norm, G_loc_norm), torch.cat(C1, C2))



        L_clsloc.backward()

