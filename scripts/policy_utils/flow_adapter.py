import torch

class DiffusionAsFlow:
    """
    Wraps your diffusion UNet as a flow: exposes velocity v(z_t, τ).
    Works for pred_type in {'epsilon','sample'} (diffusers' 'v_prediction' not used here).
    """
    def __init__(self, model, noise_scheduler, pred_type: str):
        self.model = model
        self.scheduler = noise_scheduler
        assert pred_type in ("epsilon", "sample")
        self.pred_type = pred_type

        # Precompute the sqrt(ᾱ_t) tables on the right device at runtime
        # We'll index with the scheduler's discrete timestep ids (same ones you already pass to UNet).
    def _eps_x0_from_model(self, z_t, t_id, global_cond):
        """
        z_t: [B,H,D] current noisy actions (normalized space)
        t_id: scalar int or 1D tensor of ints from scheduler.timesteps
        """
        B = z_t.shape[0]

        # batch timestep handling
        if torch.is_tensor(t_id):
            if t_id.ndim == 0:
                t_ids = t_id.to(dtype=torch.long, device=z_t.device).expand(B)
            else:
                assert t_id.shape[0] == B, "t_id must be scalar or shape [B]"
                t_ids = t_id.to(dtype=torch.long, device=z_t.device)
        else:
            t_ids = torch.full((B,), int(t_id), dtype=torch.long, device=z_t.device)
        
        # 1) UNet forward at this timestep
        model_out = self.model(z_t, t_ids, local_cond=None, global_cond=global_cond)

        # 2) Convert to (ε̂, x̂0) using the scheduler's ᾱ_t
        #    We fetch ᾱ_t with the SAME discrete id 't_id' you're already using.
        # if isinstance(t_id, torch.Tensor):
        #     t_int = int(t_id[0].item())
        # else:
        #     t_int = int(t_id)

        # alphas_cumprod = self.scheduler.alphas_cumprod.to(z_t.device).to(z_t.dtype)
        # a_bar = alphas_cumprod[t_int].view(1,1,1)
        # sqrt_ab = torch.sqrt(a_bar)
        # # sqrt_one_minus_ab = torch.sqrt(1.0 - a_bar + 1e-12)
        # sqrt_one_minus_ab = torch.sqrt(1.0 - a_bar)

        alphas_cumprod = self.scheduler.alphas_cumprod.to(z_t.device).to(z_t.dtype)
        a_bar = alphas_cumprod.index_select(0, t_ids).view(B, 1, 1)
        sqrt_ab = torch.sqrt(a_bar)
        sqrt_one_minus_ab = torch.sqrt(1.0 - a_bar)

        if self.pred_type == 'epsilon':         # model predicts ε̂
            eps_hat = model_out
            x0_hat = (z_t - sqrt_one_minus_ab * eps_hat) / (sqrt_ab)
        else:                                   # 'sample' — model predicts x̂0
            x0_hat = model_out
            eps_hat = (z_t - sqrt_ab * x0_hat) / (sqrt_one_minus_ab)

        return eps_hat, x0_hat

    def velocity(self, z_t, t_id, global_cond):
        eps_hat, x0_hat = self._eps_x0_from_model(z_t, t_id, global_cond)
        return eps_hat - x0_hat  # the flow velocity field