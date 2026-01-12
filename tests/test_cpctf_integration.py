import torch

from models.HtulTS import Model


class DummyArgs:
    def __init__(self):
        self.input_len = 64
        self.pred_len = None
        self.task_name = 'pretrain'
        self.use_norm = False
        self.use_noise = False
        self.enc_in = 1
        self.num_classes = 2
        # CPCTF specific args
        self.use_cpc = True
        self.cpc_freq_mask_ratio = 0.2
        self.cpc_time_mask_ratio = 0.25
        self.cpc_lambda = 0.3
        self.cpc_use_learned_mask = True
        self.cpc_loss_type = 'l2'


def test_cpctf_integration_forward():
    args = DummyArgs()
    model = Model(args)
    model.model.train()

    B = 2
    x = torch.randn(B, args.input_len, args.enc_in)

    recon, loss = model.pretrain(x)

    # Reconstruction shape should match input
    assert recon.shape == x.shape

    # Loss should be a tensor or numeric and finite
    assert torch.isfinite(loss).all()

    # last_loss_dict should exist and contain CPC entries
    last = model.model.last_loss_dict
    assert 'loss_time_to_freq' in last and 'loss_freq_to_time' in last
    assert 'loss_total' in last
    assert torch.isfinite(last['loss_time_to_freq']) or float(last['loss_time_to_freq']) == 0.0
    assert torch.isfinite(last['loss_freq_to_time']) or float(last['loss_freq_to_time']) == 0.0
